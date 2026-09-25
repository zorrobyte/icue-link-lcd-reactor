/*
 * brrr_classic: the first, plain version of brrr. A pixel-art llama on a hamster wheel for the iCUE LINK AIO pump LCD.
 *
 * The llama runs and the wheel spins at your real tokens/sec. "GPU GO BRRR" gains
 * an R for every 150 tok/s. Warm GPUs make the llama sweat; past 800 W the flames
 * come out and the caption turns to THIS IS FINE. Idle, it falls asleep: wen prompt?
 *
 * Run with --demo to cycle through every mood, --bench to write preview PNGs.
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <cairo/cairo.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/hidraw.h>
#include <math.h>
#include <netinet/in.h>
#include <nvml.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <time.h>
#include <turbojpeg.h>
#include <unistd.h>

#define SIZE            480
#define FPS             20
#define REPORT_SIZE     1024
#define HEADER_SIZE     8
#define CHUNK_SIZE      (REPORT_SIZE - HEADER_SIZE)
#define JPEG_QUALITY    85
#define POWER_MAX       1150.0      /* both cards near their 575 W limits */
#define N_GPUS          2


static const char  *gpu_bus[N_GPUS]  = { "00000000:01:00.0", "00000000:03:00.0" };  /* ZOTAC, TUF */
static const int    vllm_ports[]     = { 18090, 18091 };
#define N_PORTS     (int)(sizeof(vllm_ports) / sizeof(vllm_ports[0]))

typedef struct { double r, g, b; } rgb;

static const rgb BLUE   = { 61 / 255.0, 174 / 255.0, 233 / 255.0 };
static const rgb ORANGE = { 233 / 255.0, 120 / 255.0, 61 / 255.0 };
static const rgb WHITE  = { 240 / 255.0, 240 / 255.0, 245 / 255.0 };
static const rgb DIM    = { 110 / 255.0, 115 / 255.0, 125 / 255.0 };
static const rgb BG     = { 6 / 255.0, 7 / 255.0, 12 / 255.0 };

typedef struct {
    double load[N_GPUS];        /* 0..1 */
    double power[N_GPUS];       /* W */
    int    temp[N_GPUS];        /* C */
    double tok_s;               /* all servers */
    double tok_port[2];         /* per server: [0] ZOTAC's vLLM, [1] TUF's vLLM */
    int    running;
} stats;

static volatile sig_atomic_t stop;
static int demo;

static void on_signal(int sig) { (void)sig; stop = 1; }

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static double clamp01(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

static rgb lerp(rgb a, rgb b, double t)
{
    rgb c = { a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t };
    return c;
}

/* cool cyan -> amber -> hot red as power rises */
__attribute__((unused)) static rgb heat_color(double t)
{
    static const rgb c0 = { 40 / 255.0, 200 / 255.0, 1.0 };
    static const rgb c1 = { 1.0, 170 / 255.0, 40 / 255.0 };
    static const rgb c2 = { 1.0, 40 / 255.0, 30 / 255.0 };
    t = clamp01(t);
    return t <= 0.55 ? lerp(c0, c1, t / 0.55) : lerp(c1, c2, (t - 0.55) / 0.45);
}

/* ---------------------------------------------------------------- LCD */

/*
 * Set LCD_DUMP_DIR to write every frame as a JPEG into that directory instead of
 * sending it to the pump (used to make the README GIFs; no device needed).
 */
static const char *lcd_dump_dir(void)
{
    const char *d = getenv("LCD_DUMP_DIR");
    return d && *d ? d : NULL;
}

static int lcd_open(void)
{
    if (lcd_dump_dir())
        return open("/dev/null", O_WRONLY | O_CLOEXEC);

    DIR *dir = opendir("/sys/class/hidraw");
    struct dirent *e;
    char path[512], buf[1024];
    int fd = -1;

    if (!dir)
        return -1;
    while ((e = readdir(dir)) && fd < 0) {
        if (strncmp(e->d_name, "hidraw", 6) != 0)
            continue;
        snprintf(path, sizeof(path), "/sys/class/hidraw/%s/device/uevent", e->d_name);
        FILE *f = fopen(path, "r");
        if (!f)
            continue;
        size_t n = fread(buf, 1, sizeof(buf) - 1, f);
        fclose(f);
        buf[n] = 0;
        if (strcasestr(buf, "00001B1C:00000C4E")) {
            snprintf(path, sizeof(path), "/dev/%s", e->d_name);
            fd = open(path, O_RDWR | O_CLOEXEC);
        }
    }
    closedir(dir);
    return fd;
}

static void lcd_brightness(int fd, int percent)
{
    if (lcd_dump_dir())
        return;
    unsigned char rep[4] = { 0x03, 0x0B, (unsigned char)percent, 0x01 };
    ioctl(fd, HIDIOCSFEATURE(sizeof(rep)), rep);
}

static int lcd_send(int fd, const unsigned char *jpeg, unsigned long len)
{
    if (lcd_dump_dir()) {
        static int frame;
        char path[512];
        snprintf(path, sizeof(path), "%s/frame_%05d.jpg", lcd_dump_dir(), frame++);
        FILE *f = fopen(path, "wb");
        if (f) {
            fwrite(jpeg, 1, len, f);
            fclose(f);
        }
        return 0;
    }

    unsigned char rep[REPORT_SIZE];
    unsigned long off = 0;
    int idx = 0;

    while (off < len) {
        unsigned long n = len - off > CHUNK_SIZE ? CHUNK_SIZE : len - off;
        memset(rep, 0, sizeof(rep));
        rep[0] = 0x02;
        rep[1] = 0x05;
        rep[2] = 0x01;
        rep[3] = (off + n == len) ? 0x01 : 0x00;   /* last chunk tells the panel to render */
        rep[4] = (unsigned char)idx++;
        rep[6] = n & 0xFF;
        rep[7] = (n >> 8) & 0xFF;
        memcpy(rep + HEADER_SIZE, jpeg + off, n);
        if (write(fd, rep, sizeof(rep)) != (ssize_t)sizeof(rep))
            return -1;
        off += n;
    }
    return 0;
}

/* ---------------------------------------------------------------- data */

static nvmlDevice_t nvml_dev[N_GPUS];
static int nvml_ok;

static void gpus_init(void)
{
    if (nvmlInit_v2() != NVML_SUCCESS)
        return;
    nvml_ok = 1;
    for (int i = 0; i < N_GPUS; i++)
        if (nvmlDeviceGetHandleByPciBusId_v2(gpu_bus[i], &nvml_dev[i]) != NVML_SUCCESS)
            nvml_dev[i] = NULL;
}

static void gpus_poll(stats *s)
{
    if (!nvml_ok)
        return;
    for (int i = 0; i < N_GPUS; i++) {
        nvmlUtilization_t u;
        nvmlTemperature_t temp = { .version = nvmlTemperature_v1, .sensorType = NVML_TEMPERATURE_GPU };
        unsigned int mw;
        if (!nvml_dev[i])
            continue;
        if (nvmlDeviceGetUtilizationRates(nvml_dev[i], &u) == NVML_SUCCESS)
            s->load[i] = u.gpu / 100.0;
        if (nvmlDeviceGetPowerUsage(nvml_dev[i], &mw) == NVML_SUCCESS)
            s->power[i] = mw / 1000.0;
        if (nvmlDeviceGetTemperatureV(nvml_dev[i], &temp) == NVML_SUCCESS)
            s->temp[i] = temp.temperature;
    }
}

/* Fetch http://127.0.0.1:port/metrics into buf. Returns bytes read or -1. */
static long http_metrics(int port, char *buf, long cap)
{
    struct sockaddr_in addr = { .sin_family = AF_INET, .sin_port = htons(port) };
    struct timeval tv = { .tv_sec = 0, .tv_usec = 300000 };
    static const char req[] = "GET /metrics HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n";
    long got = 0;
    int fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);

    if (fd < 0)
        return -1;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0 ||
        write(fd, req, sizeof(req) - 1) != (ssize_t)(sizeof(req) - 1)) {
        close(fd);
        return -1;
    }
    while (got < cap - 1) {
        ssize_t n = read(fd, buf + got, cap - 1 - got);
        if (n <= 0)
            break;
        got += n;
    }
    close(fd);
    buf[got] = 0;
    return got;
}

/* Sum every sample of a metric family, e.g. "vllm:generation_tokens_total". */
static double metric_sum(const char *text, const char *name)
{
    size_t nlen = strlen(name);
    double sum = 0;
    const char *p = text;

    while ((p = strstr(p, name))) {
        if ((p == text || p[-1] == '\n') && (p[nlen] == '{' || p[nlen] == ' ')) {
            const char *eol = strchr(p, '\n');
            const char *sp = p + nlen;
            if (*sp == '{')
                sp = strchr(sp, '}');
            if (sp && (!eol || sp < eol))
                sum += strtod(sp + 1, NULL);
        }
        p += nlen;
    }
    return sum;
}

static void vllm_poll(stats *s, double t)
{
    static char *buf;
    static double last_tokens[N_PORTS], last_t[N_PORTS];
    static int have[N_PORTS];
    int running = 0;

    if (!buf && !(buf = malloc(1 << 20)))
        return;
    s->tok_s = 0;
    for (int i = 0; i < N_PORTS; i++) {
        double tokens, rate = 0;
        if (http_metrics(vllm_ports[i], buf, 1 << 20) <= 0) {
            have[i] = 0;
            s->tok_port[i] *= 0.5;
            s->tok_s += s->tok_port[i];
            continue;
        }
        tokens   = metric_sum(buf, "vllm:generation_tokens_total");
        running += (int)metric_sum(buf, "vllm:num_requests_running");
        if (have[i] && tokens >= last_tokens[i] && t > last_t[i])
            rate = (tokens - last_tokens[i]) / (t - last_t[i]);
        s->tok_port[i] = 0.6 * s->tok_port[i] + 0.4 * rate;
        s->tok_s += s->tok_port[i];
        last_tokens[i] = tokens;
        last_t[i] = t;
        have[i] = 1;
    }
    s->running = running;
}

static void demo_poll(stats *s, double t)
{
    double busy = 0.5 + 0.5 * sin(t * 0.25);
    s->load[0]  = clamp01(busy + (rand() / (double)RAND_MAX - 0.5) * 0.1);
    s->load[1]  = clamp01(busy * 0.9 + (rand() / (double)RAND_MAX - 0.5) * 0.1);
    s->power[0] = 25 + 420 * s->load[0];
    s->power[1] = 30 + 410 * s->load[1];
    s->temp[0]  = (int)(35 + 20 * busy);
    s->temp[1]  = (int)(42 + 30 * busy);
    s->tok_port[0] = 0.8 * s->tok_port[0] + 0.2 * (busy > 0.15 ? 140 * busy : 0);
    s->tok_port[1] = 0.8 * s->tok_port[1] + 0.2 * (busy > 0.25 ? 120 * (busy - 0.1) : 0);
    s->tok_s    = s->tok_port[0] + s->tok_port[1];
    s->running  = (int)(busy * 6);
}

/* ---------------------------------------------------------------- sprite */

#define SPR_W           24
#define SPR_H           24
#define PX              6           /* screen pixels per sprite pixel */
#define WHEEL_CX        240.0
#define WHEEL_CY        262.0
#define WHEEL_R         148.0
#define WHEEL_RIM       12.0

/* Side view llama, facing right. Legs differ between the two run frames. */
static const char *llama_top[18] = {
    "................W..W....",
    "................WW.WW...",
    "................WWWWW...",
    "...............WWWWWWW..",
    "...............WWWKWWWW.",
    "...............WWWWWWWWN",
    "...............WWWWWWWW.",
    "................WWWWWW..",
    "................WWWWW...",
    "................WWWWW...",
    "................WWWWW...",
    "................WWWWS...",
    "..WW............WWWWS...",
    ".WWWWWWWWWWWWWWWWWWWS...",
    ".WWWWWWWWWWWWWWWWWWWS...",
    "..WWWWWWWWWWWWWWWWWWS...",
    "..WWWWWWWWWWWWWWWWWSS...",
    "..SWWWWWWWWWWWWWWWSS....",
};
static const char *legs_stride[6] = {
    "...SSWW.........WSS.....",
    "...WW.WW........WW.WW...",
    "..WW...WW......WW...WW..",
    ".WW.....WW....WW.....WW.",
    ".KK.....KK....KK.....KK.",
    "........................",
};
static const char *legs_gather[6] = {
    "...SSWW........WWSS.....",
    "....WWW.........WWW.....",
    "....WWW.........WWW.....",
    "....WWW.........WWW.....",
    "....KKK.........KKK.....",
    "........................",
};

static const rgb WOOL  = { 242 / 255.0, 232 / 255.0, 212 / 255.0 };
static const rgb SHADE = { 205 / 255.0, 188 / 255.0, 160 / 255.0 };
static const rgb INK   = { 25 / 255.0, 20 / 255.0, 20 / 255.0 };
static const rgb NOSE  = { 90 / 255.0, 60 / 255.0, 55 / 255.0 };
static const rgb PINK  = { 240 / 255.0, 120 / 255.0, 140 / 255.0 };
static const rgb SWEAT = { 120 / 255.0, 200 / 255.0, 255 / 255.0 };

static int check_sprites(void)
{
    for (int i = 0; i < 18; i++)
        if (strlen(llama_top[i]) != SPR_W)
            return -1;
    for (int i = 0; i < 6; i++)
        if (strlen(legs_stride[i]) != SPR_W || strlen(legs_gather[i]) != SPR_W)
            return -1;
    return 0;
}

static void set_rgb(cairo_t *cr, rgb c) { cairo_set_source_rgb(cr, c.r, c.g, c.b); }

static void draw_llama(cairo_t *cr, double x0, double y0, int stride, int asleep, int tongue)
{
    cairo_set_antialias(cr, CAIRO_ANTIALIAS_NONE);
    for (int row = 0; row < SPR_H; row++) {
        const char *line = row < 18 ? llama_top[row] : (stride ? legs_stride : legs_gather)[row - 18];
        for (int col = 0; col < SPR_W; col++) {
            char ch = line[col];
            if (row == 4 && col == 18 && asleep)
                ch = 'S';                               /* eyes closed */
            if (row == 6 && col == 23 && tongue)
                ch = 'P';                               /* tongue out at high load */
            rgb c;
            switch (ch) {
            case 'W': c = WOOL; break;
            case 'S': c = SHADE; break;
            case 'K': c = INK; break;
            case 'N': c = NOSE; break;
            case 'P': c = PINK; break;
            default: continue;
            }
            set_rgb(cr, c);
            cairo_rectangle(cr, x0 + col * PX, y0 + row * PX, PX, PX);
            cairo_fill(cr);
        }
    }
    cairo_set_antialias(cr, CAIRO_ANTIALIAS_DEFAULT);
}

/* ---------------------------------------------------------------- effects */

#define MAX_FX          96

typedef struct { double x, y, vx, vy, life; int kind; } fx;   /* kind 0 sweat, 1 zzz, 2 speed line */

static fx     fxs[MAX_FX];
static double wheel_angle, stride_phase, fx_acc[3];

static double frand(void) { return rand() / (double)RAND_MAX; }

static void fx_spawn(int kind, double x, double y, double vx, double vy, double life)
{
    for (int i = 0; i < MAX_FX; i++)
        if (fxs[i].life <= 0) {
            fxs[i] = (fx){ x, y, vx, vy, life, kind };
            return;
        }
}

#define HOT_TOK         1200.0      /* flames and THIS IS FINE from here */
#define SWEAT_TOK       500.0       /* the llama starts sweating from here */

static int hot(double tok) { return tok >= HOT_TOK; }

static void simulate(const stats *s, const double tok, double dt, double llama_x, double llama_y)
{
    int asleep = s->tok_s < 1 && s->running == 0;
    double speed = asleep ? 0 : 1.5 + fmin(tok, 2400) / 160;    /* strides per second */

    stride_phase += speed * dt;
    wheel_angle -= speed * 0.55 * dt;                             /* wheel turns under the runner */

    /* Sweat once throughput gets serious, more the harder it goes */
    fx_acc[0] += (tok >= SWEAT_TOK && !asleep ? 2 + (tok - SWEAT_TOK) / 120 : 0) * dt;
    while (fx_acc[0] >= 1) {
        fx_spawn(0, llama_x + 17 * PX, llama_y + 2 * PX, -60 - frand() * 60, -80 - frand() * 60, 0.9);
        fx_acc[0] -= 1;
    }
    /* Z's while asleep */
    fx_acc[1] += (asleep ? 0.7 : 0) * dt;
    while (fx_acc[1] >= 1) {
        fx_spawn(1, llama_x + 20 * PX, llama_y + 1 * PX, 18, -32, 3.0);
        fx_acc[1] -= 1;
    }
    /* Speed lines behind the llama when it's really going */
    fx_acc[2] += (tok > 300 && !asleep ? fmin(tok, 2400) / 60 : 0) * dt;
    while (fx_acc[2] >= 1) {
        fx_spawn(2, llama_x + 2 * PX, llama_y + (8 + frand() * 12) * PX, -420 - frand() * 200, 0, 0.35);
        fx_acc[2] -= 1;
    }

    for (int i = 0; i < MAX_FX; i++) {
        fx *f = &fxs[i];
        if (f->life <= 0)
            continue;
        f->x += f->vx * dt;
        f->y += f->vy * dt;
        if (f->kind == 0)
            f->vy += 380 * dt;                                    /* droplets fall */
        f->life -= dt;
    }
}

/* ---------------------------------------------------------------- render */

static void text_center(cairo_t *cr, double x, double y, double size, int bold, rgb c, const char *s)
{
    cairo_text_extents_t ext;
    cairo_select_font_face(cr, "DejaVu Sans Mono", CAIRO_FONT_SLANT_NORMAL,
                           bold ? CAIRO_FONT_WEIGHT_BOLD : CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, size);
    cairo_text_extents(cr, s, &ext);
    cairo_move_to(cr, x - ext.width / 2 - ext.x_bearing, y - ext.height / 2 - ext.y_bearing);
    set_rgb(cr, c);
    cairo_show_text(cr, s);
}

/* Meme caption: white Impact-style text with a thick black outline, shrunk to fit */
static void meme_text(cairo_t *cr, double x, double y, double size, double max_w, rgb fill, const char *s)
{
    cairo_text_extents_t ext;
    cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, size);
    cairo_text_extents(cr, s, &ext);
    if (ext.width > max_w) {
        size *= max_w / ext.width;
        cairo_set_font_size(cr, size);
        cairo_text_extents(cr, s, &ext);
    }
    cairo_move_to(cr, x - ext.width / 2 - ext.x_bearing, y - ext.height / 2 - ext.y_bearing);
    cairo_text_path(cr, s);
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
    cairo_set_line_width(cr, size * 0.16);
    cairo_set_source_rgb(cr, 0, 0, 0);
    cairo_stroke_preserve(cr);
    set_rgb(cr, fill);
    cairo_fill(cr);
}

typedef struct { double zotac, tuf, tok; } shown_t;

static void draw_wheel(cairo_t *cr)
{
    const rgb METAL = { 0.36, 0.38, 0.44 }, METAL_HI = { 0.62, 0.65, 0.72 };

    /* Stand */
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    cairo_set_line_width(cr, 10);
    set_rgb(cr, METAL);
    cairo_move_to(cr, WHEEL_CX, WHEEL_CY);
    cairo_line_to(cr, WHEEL_CX - 95, WHEEL_CY + 205);
    cairo_move_to(cr, WHEEL_CX, WHEEL_CY);
    cairo_line_to(cr, WHEEL_CX + 95, WHEEL_CY + 205);
    cairo_stroke(cr);

    /* Spokes */
    cairo_set_line_width(cr, 3);
    cairo_set_source_rgba(cr, METAL.r, METAL.g, METAL.b, 0.55);
    for (int i = 0; i < 10; i++) {
        double a = wheel_angle + i * 2 * M_PI / 10;
        cairo_move_to(cr, WHEEL_CX, WHEEL_CY);
        cairo_line_to(cr, WHEEL_CX + (WHEEL_R - WHEEL_RIM) * cos(a), WHEEL_CY + (WHEEL_R - WHEEL_RIM) * sin(a));
    }
    cairo_stroke(cr);

    /* Rim, with rungs that show it turning */
    cairo_set_line_width(cr, WHEEL_RIM);
    set_rgb(cr, METAL);
    cairo_new_sub_path(cr);
    cairo_arc(cr, WHEEL_CX, WHEEL_CY, WHEEL_R - WHEEL_RIM / 2, 0, 2 * M_PI);
    cairo_stroke(cr);
    cairo_set_line_width(cr, 3);
    set_rgb(cr, METAL_HI);
    for (int i = 0; i < 36; i++) {
        double a = wheel_angle + i * 2 * M_PI / 36;
        double r0 = WHEEL_R - WHEEL_RIM + 2, r1 = WHEEL_R - 2;
        cairo_move_to(cr, WHEEL_CX + r0 * cos(a), WHEEL_CY + r0 * sin(a));
        cairo_line_to(cr, WHEEL_CX + r1 * cos(a), WHEEL_CY + r1 * sin(a));
    }
    cairo_stroke(cr);

    /* Hub */
    set_rgb(cr, METAL_HI);
    cairo_arc(cr, WHEEL_CX, WHEEL_CY, 9, 0, 2 * M_PI);
    cairo_fill(cr);
}

/* Blocky pixel flames along the bottom of the screen */
static void draw_flames(cairo_t *cr, double t, double intensity)
{
    const rgb F1 = { 1.0, 0.85, 0.2 }, F2 = { 1.0, 0.5, 0.1 }, F3 = { 0.85, 0.15, 0.05 };
    const int cell = 8;
    cairo_set_antialias(cr, CAIRO_ANTIALIAS_NONE);
    for (int x = 60; x < SIZE - 60; x += cell) {
        double n = 0.5 + 0.5 * sin(x * 0.11 + t * 9) * sin(x * 0.037 - t * 5.3);
        int h = (int)((2 + 7 * n) * intensity);
        for (int k = 0; k < h; k++) {
            double y = SIZE - 18 - (k + 1) * cell;
            rgb c = k < h * 0.35 ? F3 : k < h * 0.7 ? F2 : F1;
            set_rgb(cr, c);
            cairo_rectangle(cr, x, y, cell - 1, cell - 1);
            cairo_fill(cr);
        }
    }
    cairo_set_antialias(cr, CAIRO_ANTIALIAS_DEFAULT);
}

static void render(cairo_t *cr, const stats *s, const shown_t *sh, double t)
{
    const double c = SIZE / 2.0;
    double total_w = s->power[0] + s->power[1];
    int asleep = s->tok_s < 1 && s->running == 0;
    int is_hot = hot(sh->tok);
    double llama_x = WHEEL_CX - SPR_W * PX / 2.0 + 6;
    double llama_y = WHEEL_CY + WHEEL_R - WHEEL_RIM - 23 * PX;
    int stride = !asleep && ((int)(stride_phase * 2) & 1);
    char txt[96];

    /* Background: dark, warming toward red as throughput climbs */
    {
        double heat = clamp01((sh->tok - 300) / 1500);
        cairo_pattern_t *g = cairo_pattern_create_radial(c, c + 60, 40, c, c + 60, 300);
        cairo_pattern_add_color_stop_rgb(g, 0, 0.08 + 0.25 * heat, 0.09, 0.14 - 0.08 * heat);
        cairo_pattern_add_color_stop_rgb(g, 1, BG.r, BG.g, BG.b);
        cairo_set_source(cr, g);
        cairo_paint(cr);
        cairo_pattern_destroy(g);
    }

    if (is_hot)
        draw_flames(cr, t, clamp01((sh->tok - HOT_TOK) / 800) * 0.7 + 0.3);

    draw_wheel(cr);

    /* Speed lines behind the runner */
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    for (int i = 0; i < MAX_FX; i++) {
        const fx *f = &fxs[i];
        if (f->life <= 0 || f->kind != 2)
            continue;
        cairo_set_source_rgba(cr, 1, 1, 1, clamp01(f->life / 0.35) * 0.7);
        cairo_set_line_width(cr, 3);
        cairo_move_to(cr, f->x, f->y);
        cairo_line_to(cr, f->x + 34, f->y);
        cairo_stroke(cr);
    }

    /* The llama bobs a pixel on each stride */
    draw_llama(cr, llama_x, llama_y + (stride ? -PX : 0), stride, asleep, sh->tok > 300);

    /* Sweat and Z's */
    for (int i = 0; i < MAX_FX; i++) {
        const fx *f = &fxs[i];
        if (f->life <= 0)
            continue;
        if (f->kind == 0) {
            cairo_set_source_rgba(cr, SWEAT.r, SWEAT.g, SWEAT.b, clamp01(f->life / 0.4));
            cairo_rectangle(cr, f->x, f->y, 6, 8);
            cairo_fill(cr);
        } else if (f->kind == 1) {
            double age = 3.0 - f->life;
            cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
            cairo_set_font_size(cr, 16 + age * 9);
            cairo_move_to(cr, f->x, f->y);
            cairo_set_source_rgba(cr, 0.8, 0.85, 1.0, clamp01(f->life / 1.0));
            cairo_show_text(cr, "Z");
        }
    }

    /* Tokens/sec inside the wheel */
    if (asleep) {
        text_center(cr, c, 190, 30, 1, DIM, "0 tok/s");
    } else {
        snprintf(txt, sizeof(txt), "%.0f", sh->tok);
        meme_text(cr, c, 180, 64, 250, WHITE, txt);
        text_center(cr, c, 226, 20, 1, (rgb){ 0.9, 0.9, 0.95 }, "tok/s");
    }

    /* Top caption */
    if (asleep) {
        meme_text(cr, c, 76, 44, 320, WHITE, "wen prompt?");
    } else if (is_hot && fmod(t, 6) >= 3) {
        meme_text(cr, c, 76, 44, 320, (rgb){ 1.0, 0.6, 0.2 }, "THIS IS FINE");
    } else {
        int rs = 2 + (int)fmin(12, sh->tok / 150);
        char brr[32] = "BR";
        for (int i = 0; i < rs && i < 28; i++)
            strcat(brr, "R");
        snprintf(txt, sizeof(txt), "GPU GO %s", sh->tok < 300 ? "brr" : brr);
        meme_text(cr, c, 76, 44, 330, WHITE, txt);
    }

    /* Bottom stats: each server's tok/s either side of the total watts */
    snprintf(txt, sizeof(txt), "%.0fW", total_w);
    meme_text(cr, c, 440, 26, 110, WHITE, txt);
    snprintf(txt, sizeof(txt), "%.0f", s->tok_port[0]);
    meme_text(cr, c - 100, 430, 26, 90, BLUE, txt);
    snprintf(txt, sizeof(txt), "%.0f", s->tok_port[1]);
    meme_text(cr, c + 100, 430, 26, 90, ORANGE, txt);
}

/* ---------------------------------------------------------------- main */

int main(int argc, char **argv)
{
    cairo_surface_t *surf;
    cairo_t *cr;
    tjhandle tj;
    unsigned char *jpeg = NULL;
    stats s = { 0 };
    shown_t sh = { 0 };
    double last, next_poll = 0, t0;
    int fd = -1, bench = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--demo"))
            demo = 1;
        else if (!strcmp(argv[i], "--bench"))
            bench = 1;
    }
    if (check_sprites() < 0) {
        fprintf(stderr, "sprite rows must be %d wide\n", SPR_W);
        return 1;
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);
    srand((unsigned)time(NULL));

    surf = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cr = cairo_create(surf);
    tj = tj3Init(TJINIT_COMPRESS);
    tj3Set(tj, TJPARAM_QUALITY, JPEG_QUALITY);
    tj3Set(tj, TJPARAM_SUBSAMP, TJSAMP_420);

    const double llama_x = WHEEL_CX - SPR_W * PX / 2.0 + 6;
    const double llama_y = WHEEL_CY + WHEEL_R - WHEEL_RIM - 23 * PX;

    if (bench) {
        /* Three scenes: busy and hot, moderate, asleep */
        struct { double w0, w1, tok; int t0, t1, run; const char *png; } scenes[] = {
            { 470, 460, 1607, 74, 83, 8, "brrr_preview.png" },
            { 300, 280, 420, 55, 64, 2, "brrr_mid.png" },
            { 25, 30, 0, 36, 42, 0, "brrr_idle.png" },
        };
        for (int k = 0; k < 3; k++) {
            memset(fxs, 0, sizeof(fxs));
            s.power[0] = scenes[k].w0; s.power[1] = scenes[k].w1;
            s.temp[0] = scenes[k].t0; s.temp[1] = scenes[k].t1;
            s.tok_s = scenes[k].tok; s.tok_port[0] = scenes[k].tok * 0.45; s.tok_port[1] = scenes[k].tok * 0.55;
            s.running = scenes[k].run;
            sh.tok = scenes[k].tok;
            double b0 = now_s();
            for (int i = 0; i < 60; i++) {
                simulate(&s, sh.tok, 1.0 / FPS, llama_x, llama_y);
                render(cr, &s, &sh, 1.0 + i / (double)FPS);
                cairo_surface_flush(surf);
                size_t len = 0;
                tj3Compress8(tj, cairo_image_surface_get_data(surf), SIZE, cairo_image_surface_get_stride(surf),
                             SIZE, TJPF_BGRX, &jpeg, &len);
            }
            printf("%s: %.2f ms/frame\n", scenes[k].png, (now_s() - b0) * 1000 / 60);
            cairo_surface_write_to_png(surf, scenes[k].png);
        }
        return 0;
    }

    if (!demo)
        gpus_init();

    while ((fd = lcd_open()) < 0 && !stop)
        sleep(2);
    if (fd >= 0)
        lcd_brightness(fd, 100);

    t0 = last = now_s();
    while (!stop) {
        double t = now_s(), dt = t - last, k;
        last = t;
        if (dt > 0.25)
            dt = 0.25;

        if (t >= next_poll) {
            next_poll = t + 1.0;
            if (demo) {
                /* scale a copy each second; scaling s itself would compound */
                static stats demo_base;
                demo_poll(&demo_base, t - t0);
                s = demo_base;
                /* demo: scale up so every mood (asleep, brrr, THIS IS FINE) shows up in one cycle */
                s.tok_port[0] *= 7;
                s.tok_port[1] *= 7;
                s.tok_s = s.tok_port[0] + s.tok_port[1];
                if (s.tok_s < 120) { s.tok_s = s.tok_port[0] = s.tok_port[1] = 0; s.running = 0; }
            } else {
                gpus_poll(&s);
                vllm_poll(&s, t);
            }
        }

        k = dt * 4 > 1 ? 1 : dt * 4;
        sh.zotac += (s.load[0] - sh.zotac) * k;
        sh.tuf   += (s.load[1] - sh.tuf) * k;
        sh.tok   += (s.tok_s - sh.tok) * k;

        simulate(&s, sh.tok, dt, llama_x, llama_y);
        render(cr, &s, &sh, t - t0);
        cairo_surface_flush(surf);
        size_t len = 0;
        if (tj3Compress8(tj, cairo_image_surface_get_data(surf), SIZE, cairo_image_surface_get_stride(surf),
                         SIZE, TJPF_BGRX, &jpeg, &len) == 0) {
            if (lcd_send(fd, jpeg, len) < 0) {
                close(fd);
                while ((fd = lcd_open()) < 0 && !stop)
                    sleep(2);
                if (fd >= 0)
                    lcd_brightness(fd, 100);
            }
        }

        double spare = 1.0 / FPS - (now_s() - t);
        if (spare > 0) {
            struct timespec ts = { 0, (long)(spare * 1e9) };
            nanosleep(&ts, NULL);
        }
    }

    tj3Free(jpeg);
    tj3Destroy(tj);
    cairo_destroy(cr);
    cairo_surface_destroy(surf);
    if (nvml_ok)
        nvmlShutdown();
    if (fd >= 0)
        close(fd);
    return 0;
}
