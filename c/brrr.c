/*
 * brrr: a synthwave pixel-art llama on a hamster wheel for the iCUE LINK AIO pump LCD.
 *
 * The llama runs and the wheel spins at your real tokens/sec. "GPU GO BRRR" gains
 * an R for every 150 tok/s. Warm GPUs make the llama sweat; past 800 W the flames
 * come out and the caption turns to THIS IS FINE. Idle, it falls asleep: wen prompt?
 *
 * --showcase runs a scripted 44 s loop of every mood at 30 fps (for filming),
 * --demo cycles simulated data, --bench writes preview PNGs.
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
__attribute__((unused)) static const rgb DIM    = { 110 / 255.0, 115 / 255.0, 125 / 255.0 };
__attribute__((unused)) static const rgb BG     = { 6 / 255.0, 7 / 255.0, 12 / 255.0 };

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

static int lcd_open(void)
{
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
    unsigned char rep[4] = { 0x03, 0x0B, (unsigned char)percent, 0x01 };
    ioctl(fd, HIDIOCSFEATURE(sizeof(rep)), rep);
}

static int lcd_send(int fd, const unsigned char *jpeg, unsigned long len)
{
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

/* ---------------------------------------------------------------- scene */

#define SPR_W           24
#define SPR_H           24
#define PX              6           /* screen pixels per sprite pixel */
#define HORIZON_Y       300.0
#define WHEEL_CX        240.0
#define WHEEL_CY        258.0
#define WHEEL_R         140.0
#define WHEEL_RIM       10.0
#define HOT_TOK         1200.0      /* flames and THIS IS FINE */
#define SHADES_TOK      800.0       /* deal with it */
#define SWEAT_TOK       500.0
#define AGI_TOK         2000.0
#define N_STARS         70
#define MAX_FX          160

#define CAPTION_FONT    "Anton"

static const rgb NEON_PINK = { 1.00, 0.16, 0.62 };
static const rgb NEON_CYAN = { 0.00, 0.90, 1.00 };
static const rgb SUN_TOP   = { 1.00, 0.83, 0.10 };
static const rgb SUN_BOT   = { 1.00, 0.16, 0.46 };

/* Side view llama, facing right */
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

typedef struct { double x, y, vx, vy, life, max_life; int kind; } fx;  /* 0 sweat, 1 Z, 2 speed line, 3 spark */

static fx     fxs[MAX_FX];
static double wheel_angle, stride_phase, grid_phase, fx_acc[4];
static double star_x[N_STARS], star_y[N_STARS], star_p[N_STARS];
static double shades_t = -1;            /* time the shades started dropping, -1 = off */
static double activity;                 /* 0 idle .. 1 flat out, eased */
static cairo_surface_t *scanlines;

static double frand(void) { return rand() / (double)RAND_MAX; }

static void set_rgb(cairo_t *cr, rgb c) { cairo_set_source_rgb(cr, c.r, c.g, c.b); }

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

static void init_scene(void)
{
    for (int i = 0; i < N_STARS; i++) {
        star_x[i] = 30 + frand() * 420;
        star_y[i] = 20 + pow(frand(), 1.4) * 200;
        star_p[i] = frand() * 2 * M_PI;
    }
    /* Faint CRT scanlines, stamped over every frame */
    scanlines = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, SIZE);
    cairo_t *sc = cairo_create(scanlines);
    cairo_set_source_rgba(sc, 0, 0, 0, 0.13);
    for (int y = 0; y < SIZE; y += 3) {
        cairo_rectangle(sc, 0, y, SIZE, 1);
    }
    cairo_fill(sc);
    cairo_destroy(sc);
}

/* Pixel colour for the llama at (row, col), after costume overlays */
static int llama_pixel(int row, int col, int stride, int asleep, int tongue, rgb *out)
{
    static const rgb WOOL = { 0.98, 0.95, 0.89 }, SHADE = { 0.84, 0.77, 0.67 }, INK = { 0.07, 0.05, 0.09 };
    static const rgb NOSE = { 0.43, 0.27, 0.27 }, TONGUE = { 1.0, 0.43, 0.59 }, BLUSH = { 1.0, 0.67, 0.75 };
    static const rgb BL_PINK = { 1.0, 0.25, 0.63 }, BL_YEL = { 1.0, 0.84, 0.25 }, BL_CYAN = { 0.0, 0.85, 1.0 };
    const char *line = row < 18 ? llama_top[row] : (stride ? legs_stride : legs_gather)[row - 18];
    char ch = line[col];

    if (ch == '.') {
        if (row == 6 && col == 23 && tongue) { *out = TONGUE; return 1; }
        return 0;
    }
    if (row == 4 && col == 18 && asleep) { *out = SHADE; return 1; }
    if (row == 5 && col == 19 && ch == 'W') { *out = BLUSH; return 1; }
    /* Saddle blanket with a fringe */
    if (col >= 6 && col <= 13 && ch == 'W') {
        if (row == 13 || row == 16) { *out = BL_PINK; return 1; }
        if (row == 14) { *out = BL_YEL; return 1; }
        if (row == 15) { *out = BL_CYAN; return 1; }
        if (row == 17 && (col & 1)) { *out = BL_YEL; return 1; }
    }
    switch (ch) {
    case 'W': *out = WOOL; return 1;
    case 'S': *out = SHADE; return 1;
    case 'K': *out = INK; return 1;
    case 'N': *out = NOSE; return 1;
    }
    return 0;
}

static void draw_llama(cairo_t *cr, double x0, double y0, int stride, int asleep, int tongue)
{
    rgb c;
    cairo_set_antialias(cr, CAIRO_ANTIALIAS_NONE);

    /* Dark outline: the silhouette offset in four directions */
    cairo_set_source_rgb(cr, 0.08, 0.02, 0.14);
    for (int row = 0; row < SPR_H; row++)
        for (int col = 0; col < SPR_W; col++)
            if (llama_pixel(row, col, stride, asleep, tongue, &c))
                cairo_rectangle(cr, x0 + col * PX - 3, y0 + row * PX - 3, PX + 6, PX + 6);
    cairo_fill(cr);

    for (int row = 0; row < SPR_H; row++)
        for (int col = 0; col < SPR_W; col++)
            if (llama_pixel(row, col, stride, asleep, tongue, &c)) {
                set_rgb(cr, c);
                cairo_rectangle(cr, x0 + col * PX, y0 + row * PX, PX, PX);
                cairo_fill(cr);
            }
    cairo_set_antialias(cr, CAIRO_ANTIALIAS_DEFAULT);
}

/* Pixel "deal with it" shades, dropping in from above */
static void draw_shades(cairo_t *cr, double x0, double y0, double drop)
{
    static const char *shades[3] = {
        "KKKKKKKK",
        "KGKKK.KK",
        ".KK...KK",
    };
    double oy = -(1 - drop) * 140;
    cairo_set_antialias(cr, CAIRO_ANTIALIAS_NONE);
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 8; c++) {
            char ch = shades[r][c];
            if (ch == '.')
                continue;
            if (ch == 'G')
                cairo_set_source_rgb(cr, 1, 1, 1);
            else
                cairo_set_source_rgb(cr, 0.02, 0.02, 0.04);
            cairo_rectangle(cr, x0 + (15 + c) * PX, y0 + (3 + r) * PX + oy, PX, PX);
            cairo_fill(cr);
        }
    cairo_set_antialias(cr, CAIRO_ANTIALIAS_DEFAULT);
}

/* ---------------------------------------------------------------- simulation */

static void fx_spawn(int kind, double x, double y, double vx, double vy, double life)
{
    for (int i = 0; i < MAX_FX; i++)
        if (fxs[i].life <= 0) {
            fxs[i] = (fx){ x, y, vx, vy, life, life, kind };
            return;
        }
}

static void simulate(const stats *s, double tok, double dt, double t, double llama_x, double llama_y)
{
    int asleep = s->tok_s < 1 && s->running == 0;
    double speed = asleep ? 0 : 1.5 + fmin(tok, 2400) / 160;     /* strides per second */

    activity += ((asleep ? 0 : clamp01(0.25 + tok / 1600)) - activity) * fmin(1, dt * 1.5);
    stride_phase += speed * dt;
    wheel_angle -= speed * 0.55 * dt;
    grid_phase = fmod(grid_phase + (0.15 + fmin(tok, 2400) / 900) * dt, 1.0);

    if (tok >= SHADES_TOK && !asleep) {
        if (shades_t < 0)
            shades_t = t;
    } else if (tok < SHADES_TOK * 0.8 || asleep) {
        shades_t = -1;
    }

    fx_acc[0] += (tok >= SWEAT_TOK && !asleep ? 2 + (tok - SWEAT_TOK) / 120 : 0) * dt;
    while (fx_acc[0] >= 1) {
        fx_spawn(0, llama_x + 16 * PX, llama_y + 2 * PX, -60 - frand() * 70, -90 - frand() * 70, 0.9);
        fx_acc[0] -= 1;
    }
    fx_acc[1] += (asleep ? 0.7 : 0) * dt;
    while (fx_acc[1] >= 1) {
        fx_spawn(1, llama_x + 20 * PX, llama_y + 1 * PX, 16, -30, 3.0);
        fx_acc[1] -= 1;
    }
    fx_acc[2] += (tok > 300 && !asleep ? fmin(tok, 2400) / 55 : 0) * dt;
    while (fx_acc[2] >= 1) {
        fx_spawn(2, llama_x + 2 * PX, llama_y + (7 + frand() * 13) * PX, -460 - frand() * 220, 0, 0.35);
        fx_acc[2] -= 1;
    }
    /* Sparks where the wheel meets the llama's feet */
    fx_acc[3] += (tok > 600 && !asleep ? fmin(tok, 2400) / 40 : 0) * dt;
    while (fx_acc[3] >= 1) {
        fx_spawn(3, WHEEL_CX - 30 + frand() * 60, WHEEL_CY + WHEEL_R - 4,
                 -120 - frand() * 220, -60 - frand() * 160, 0.5 + frand() * 0.4);
        fx_acc[3] -= 1;
    }

    for (int i = 0; i < MAX_FX; i++) {
        fx *f = &fxs[i];
        if (f->life <= 0)
            continue;
        f->x += f->vx * dt;
        f->y += f->vy * dt;
        if (f->kind == 0 || f->kind == 3)
            f->vy += (f->kind == 3 ? 520 : 380) * dt;
        f->life -= dt;
    }
}

/* ---------------------------------------------------------------- render */

static void text_path_centered(cairo_t *cr, const char *font, double x, double y, double size,
                               double max_w, const char *s, double *out_size)
{
    cairo_text_extents_t ext;
    cairo_select_font_face(cr, font, CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, size);
    cairo_text_extents(cr, s, &ext);
    if (max_w > 0 && ext.width > max_w) {
        size *= max_w / ext.width;
        cairo_set_font_size(cr, size);
        cairo_text_extents(cr, s, &ext);
    }
    cairo_new_path(cr);
    cairo_move_to(cr, x - ext.width / 2 - ext.x_bearing, y - ext.height / 2 - ext.y_bearing);
    cairo_text_path(cr, s);
    if (out_size)
        *out_size = size;
}

/* Meme text: neon glow, thick dark outline, bright fill */
static void neon_text(cairo_t *cr, double x, double y, double size, double max_w, rgb fill, rgb glow, const char *s)
{
    double sz;
    text_path_centered(cr, CAPTION_FONT, x, y, size, max_w, s, &sz);
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
    for (int k = 3; k >= 1; k--) {
        cairo_set_line_width(cr, sz * 0.12 * k + 4);
        cairo_set_source_rgba(cr, glow.r, glow.g, glow.b, 0.10);
        cairo_stroke_preserve(cr);
    }
    cairo_set_line_width(cr, sz * 0.13);
    cairo_set_source_rgb(cr, 0.06, 0.0, 0.10);
    cairo_stroke_preserve(cr);
    set_rgb(cr, fill);
    cairo_fill(cr);
}

static void draw_sky(cairo_t *cr, double t)
{
    double a = activity;
    /* Night when idle, sunset glow when busy */
    rgb top = lerp((rgb){ 0.02, 0.01, 0.07 }, (rgb){ 0.05, 0.01, 0.14 }, a);
    rgb mid = lerp((rgb){ 0.06, 0.03, 0.18 }, (rgb){ 0.30, 0.04, 0.42 }, a);
    rgb low = lerp((rgb){ 0.12, 0.05, 0.28 }, (rgb){ 0.95, 0.20, 0.45 }, a);
    cairo_pattern_t *g = cairo_pattern_create_linear(0, 0, 0, HORIZON_Y);
    cairo_pattern_add_color_stop_rgb(g, 0.0, top.r, top.g, top.b);
    cairo_pattern_add_color_stop_rgb(g, 0.6, mid.r, mid.g, mid.b);
    cairo_pattern_add_color_stop_rgb(g, 1.0, low.r, low.g, low.b);
    cairo_rectangle(cr, 0, 0, SIZE, HORIZON_Y);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);

    for (int i = 0; i < N_STARS; i++) {
        double tw = 0.5 + 0.5 * sin(t * 1.7 + star_p[i]);
        double fade = clamp01(1.1 - star_y[i] / HORIZON_Y) * (1 - 0.6 * a);
        cairo_set_source_rgba(cr, 1, 0.92, 1, (0.25 + 0.75 * tw) * fade);
        cairo_rectangle(cr, star_x[i], star_y[i], 2, 2);
        cairo_fill(cr);
    }
}

/* Striped synthwave sun that rises with throughput */
static void draw_sun(cairo_t *cr, double t)
{
    const double r = 104;
    double cy = HORIZON_Y - 60 + (1 - activity) * 120;
    double cx = WHEEL_CX;

    /* Halo */
    cairo_pattern_t *h = cairo_pattern_create_radial(cx, cy, r * 0.8, cx, cy, r * 2.0);
    cairo_pattern_add_color_stop_rgba(h, 0, SUN_BOT.r, SUN_BOT.g, SUN_BOT.b, 0.35 * (0.3 + 0.7 * activity));
    cairo_pattern_add_color_stop_rgba(h, 1, SUN_BOT.r, SUN_BOT.g, SUN_BOT.b, 0);
    cairo_rectangle(cr, 0, 0, SIZE, HORIZON_Y);
    cairo_set_source(cr, h);
    cairo_fill(cr);
    cairo_pattern_destroy(h);

    cairo_save(cr);
    cairo_rectangle(cr, 0, 0, SIZE, HORIZON_Y);
    cairo_clip(cr);
    /* Stripe cut-outs across the lower half, drifting downward */
    cairo_new_path(cr);
    cairo_arc(cr, cx, cy, r, 0, 2 * M_PI);
    cairo_set_fill_rule(cr, CAIRO_FILL_RULE_EVEN_ODD);
    double drift = fmod(t * 6, 16);
    for (int i = 0; i < 8; i++) {
        double y = cy + 4 + i * 16 + drift;
        double hgt = 1.5 + i * 1.3;
        if (y > cy + r)
            break;
        cairo_rectangle(cr, cx - r - 2, y, 2 * r + 4, hgt);
    }
    cairo_pattern_t *g = cairo_pattern_create_linear(0, cy - r, 0, cy + r);
    cairo_pattern_add_color_stop_rgb(g, 0.0, SUN_TOP.r, SUN_TOP.g, SUN_TOP.b);
    cairo_pattern_add_color_stop_rgb(g, 1.0, SUN_BOT.r, SUN_BOT.g, SUN_BOT.b);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    cairo_set_fill_rule(cr, CAIRO_FILL_RULE_WINDING);
    cairo_restore(cr);
}

static void draw_mountains(cairo_t *cr)
{
    static const double left[][2] = {
        { 0, 300 }, { 0, 238 }, { 30, 252 }, { 62, 220 }, { 96, 258 }, { 122, 244 }, { 160, 300 },
    };
    static const double right[][2] = {
        { 480, 300 }, { 480, 242 }, { 452, 228 }, { 420, 256 }, { 392, 216 }, { 356, 262 }, { 330, 300 },
    };
    const double (*polys[2])[2] = { left, right };
    for (int p = 0; p < 2; p++) {
        cairo_new_path(cr);
        for (int i = 0; i < 7; i++)
            cairo_line_to(cr, polys[p][i][0], polys[p][i][1]);
        cairo_close_path(cr);
        cairo_set_source_rgb(cr, 0.10, 0.02, 0.20);
        cairo_fill_preserve(cr);
        cairo_set_line_width(cr, 1.5);
        cairo_set_source_rgba(cr, NEON_PINK.r, NEON_PINK.g, NEON_PINK.b, 0.55);
        cairo_stroke(cr);
    }
}

/* Perspective neon grid scrolling toward the viewer */
static void draw_floor(cairo_t *cr)
{
    const double vx = WHEEL_CX, vy = HORIZON_Y;
    cairo_pattern_t *g = cairo_pattern_create_linear(0, HORIZON_Y, 0, SIZE);
    cairo_pattern_add_color_stop_rgb(g, 0, 0.10, 0.01, 0.20);
    cairo_pattern_add_color_stop_rgb(g, 1, 0.02, 0.00, 0.05);
    cairo_rectangle(cr, 0, HORIZON_Y, SIZE, SIZE - HORIZON_Y);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);

    cairo_new_path(cr);
    for (int i = -12; i <= 12; i++) {
        cairo_move_to(cr, vx + i * 10, vy);
        cairo_line_to(cr, vx + i * 70, SIZE + 40);
    }
    for (int i = 0; i < 14; i++) {
        double z = (i + grid_phase) / 14.0;          /* 0 far .. 1 near */
        double y = vy + pow(z, 2.4) * (SIZE - vy + 20);
        cairo_move_to(cr, 0, y);
        cairo_line_to(cr, SIZE, y);
    }
    cairo_set_line_width(cr, 5);
    cairo_set_source_rgba(cr, NEON_PINK.r, NEON_PINK.g, NEON_PINK.b, 0.12);
    cairo_stroke_preserve(cr);
    cairo_set_line_width(cr, 1.4);
    cairo_set_source_rgba(cr, NEON_PINK.r, NEON_PINK.g, NEON_PINK.b, 0.85);
    cairo_stroke(cr);

    /* Horizon line */
    cairo_rectangle(cr, 0, HORIZON_Y - 1, SIZE, 2);
    cairo_set_source_rgba(cr, 1, 0.6, 0.85, 0.9);
    cairo_fill(cr);
}

static void draw_flames(cairo_t *cr, double t, double intensity)
{
    const rgb F1 = { 1.0, 0.90, 0.30 }, F2 = { 1.0, 0.50, 0.10 }, F3 = { 0.90, 0.10, 0.20 };
    const int cell = 8;

    /* Glow */
    cairo_pattern_t *g = cairo_pattern_create_linear(0, SIZE - 110 * intensity - 20, 0, SIZE);
    cairo_pattern_add_color_stop_rgba(g, 0, 1, 0.35, 0.1, 0);
    cairo_pattern_add_color_stop_rgba(g, 1, 1, 0.35, 0.1, 0.45);
    cairo_rectangle(cr, 0, SIZE - 130, SIZE, 130);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);

    cairo_set_antialias(cr, CAIRO_ANTIALIAS_NONE);
    for (int x = 40; x < SIZE - 40; x += cell) {
        double n = 0.5 + 0.5 * sin(x * 0.11 + t * 9) * sin(x * 0.037 - t * 5.3);
        int h = (int)((2 + 8 * n) * intensity);
        for (int k = 0; k < h; k++) {
            double y = SIZE - 12 - (k + 1) * cell;
            rgb c = k < h * 0.35 ? F3 : k < h * 0.7 ? F2 : F1;
            set_rgb(cr, c);
            cairo_rectangle(cr, x, y, cell - 1, cell - 1);
            cairo_fill(cr);
        }
    }
    cairo_set_antialias(cr, CAIRO_ANTIALIAS_DEFAULT);
}

static void draw_wheel_stand(cairo_t *cr)
{
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    cairo_new_path(cr);
    cairo_move_to(cr, WHEEL_CX, WHEEL_CY);
    cairo_line_to(cr, WHEEL_CX - 92, WHEEL_CY + 190);
    cairo_move_to(cr, WHEEL_CX, WHEEL_CY);
    cairo_line_to(cr, WHEEL_CX + 92, WHEEL_CY + 190);
    cairo_set_line_width(cr, 12);
    cairo_set_source_rgb(cr, 0.12, 0.05, 0.22);
    cairo_stroke_preserve(cr);
    cairo_set_line_width(cr, 3);
    set_rgb(cr, NEON_CYAN);
    cairo_stroke(cr);
}

static void neon_stroke(cairo_t *cr, rgb c, double w, double a)
{
    cairo_set_line_width(cr, w * 3.2);
    cairo_set_source_rgba(cr, c.r, c.g, c.b, 0.14 * a);
    cairo_stroke_preserve(cr);
    cairo_set_line_width(cr, w);
    cairo_set_source_rgba(cr, c.r, c.g, c.b, a);
    cairo_stroke(cr);
}

static void draw_wheel(cairo_t *cr, double speed)
{
    /* Spokes, motion-blurred with trailing copies at speed */
    int ghosts = speed > 4 ? 3 : 1;
    for (int gh = ghosts - 1; gh >= 0; gh--) {
        double lag = gh * 0.06 * fmin(speed, 16) / 16;
        cairo_new_path(cr);
        for (int i = 0; i < 8; i++) {
            double a = wheel_angle + lag + i * 2 * M_PI / 8;
            cairo_move_to(cr, WHEEL_CX, WHEEL_CY);
            cairo_line_to(cr, WHEEL_CX + (WHEEL_R - WHEEL_RIM) * cos(a), WHEEL_CY + (WHEEL_R - WHEEL_RIM) * sin(a));
        }
        neon_stroke(cr, NEON_CYAN, 2.0, gh ? 0.25 : 0.7);
    }

    /* Rim */
    cairo_new_path(cr);
    cairo_arc(cr, WHEEL_CX, WHEEL_CY, WHEEL_R - WHEEL_RIM / 2, 0, 2 * M_PI);
    cairo_set_line_width(cr, WHEEL_RIM + 6);
    cairo_set_source_rgb(cr, 0.10, 0.03, 0.20);
    cairo_stroke(cr);
    cairo_new_path(cr);
    cairo_arc(cr, WHEEL_CX, WHEEL_CY, WHEEL_R - 1, 0, 2 * M_PI);
    neon_stroke(cr, NEON_CYAN, 2.5, 1);
    cairo_new_path(cr);
    cairo_arc(cr, WHEEL_CX, WHEEL_CY, WHEEL_R - WHEEL_RIM, 0, 2 * M_PI);
    neon_stroke(cr, NEON_PINK, 2.0, 0.9);
    /* Rungs */
    cairo_new_path(cr);
    for (int i = 0; i < 40; i++) {
        double a = wheel_angle + i * 2 * M_PI / 40;
        double r0 = WHEEL_R - WHEEL_RIM, r1 = WHEEL_R - 1;
        cairo_move_to(cr, WHEEL_CX + r0 * cos(a), WHEEL_CY + r0 * sin(a));
        cairo_line_to(cr, WHEEL_CX + r1 * cos(a), WHEEL_CY + r1 * sin(a));
    }
    cairo_set_line_width(cr, 1.5);
    cairo_set_source_rgba(cr, NEON_CYAN.r, NEON_CYAN.g, NEON_CYAN.b, 0.7);
    cairo_stroke(cr);

    /* Hub */
    cairo_new_path(cr);
    cairo_arc(cr, WHEEL_CX, WHEEL_CY, 8, 0, 2 * M_PI);
    cairo_set_source_rgb(cr, 0.10, 0.03, 0.20);
    cairo_fill_preserve(cr);
    neon_stroke(cr, NEON_CYAN, 2, 1);
}

typedef struct { double zotac, tuf, tok; } shown_t;

static const char *caption_override;        /* set by the showcase script, NULL = automatic */

static void render(cairo_t *cr, const stats *s, const shown_t *sh, double t)
{
    const double c = SIZE / 2.0;
    double total_w = s->power[0] + s->power[1];
    int asleep = s->tok_s < 1 && s->running == 0;
    double tok = sh->tok;
    double speed = asleep ? 0 : 1.5 + fmin(tok, 2400) / 160;
    double llama_x = WHEEL_CX - SPR_W * PX / 2.0 + 6;
    double llama_y = WHEEL_CY + WHEEL_R - WHEEL_RIM - 23 * PX;
    int stride = !asleep && ((int)(stride_phase * 2) & 1);
    char txt[96];

    draw_sky(cr, t);
    draw_sun(cr, t);
    draw_mountains(cr);
    draw_floor(cr);
    if (tok >= HOT_TOK && !asleep)
        draw_flames(cr, t, clamp01((tok - HOT_TOK) / 800) * 0.7 + 0.3);
    draw_wheel_stand(cr);

    /* Dim the scene inside the wheel slightly so the runner pops */
    cairo_new_path(cr);
    cairo_arc(cr, WHEEL_CX, WHEEL_CY, WHEEL_R - WHEEL_RIM, 0, 2 * M_PI);
    cairo_set_source_rgba(cr, 0.05, 0.0, 0.10, 0.28);
    cairo_fill(cr);

    draw_wheel(cr, speed);

    /* Speed lines */
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    for (int i = 0; i < MAX_FX; i++) {
        const fx *f = &fxs[i];
        if (f->life <= 0 || f->kind != 2)
            continue;
        cairo_new_path(cr);
        cairo_move_to(cr, f->x, f->y);
        cairo_line_to(cr, f->x + 36, f->y);
        neon_stroke(cr, (i & 1) ? NEON_CYAN : (rgb){ 1, 1, 1 }, 2.2, clamp01(f->life / f->max_life) * 0.8);
    }

    draw_llama(cr, llama_x, llama_y + (stride ? -PX : 0), stride, asleep, tok > 300);
    if (shades_t >= 0)
        draw_shades(cr, llama_x, llama_y + (stride ? -PX : 0), clamp01((t - shades_t) / 0.7));

    /* Sweat, Z's and sparks */
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    for (int i = 0; i < MAX_FX; i++) {
        const fx *f = &fxs[i];
        if (f->life <= 0)
            continue;
        double a = clamp01(f->life / f->max_life);
        if (f->kind == 0) {
            cairo_set_source_rgba(cr, 0.55, 0.85, 1.0, a);
            cairo_rectangle(cr, f->x, f->y, 6, 8);
            cairo_fill(cr);
        } else if (f->kind == 1) {
            double age = f->max_life - f->life;
            cairo_select_font_face(cr, CAPTION_FONT, CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
            cairo_set_font_size(cr, 18 + age * 10);
            cairo_new_path(cr);
            cairo_move_to(cr, f->x, f->y);
            cairo_set_source_rgba(cr, 0.75, 0.85, 1.0, a);
            cairo_show_text(cr, "Z");
        } else if (f->kind == 3) {
            cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
            cairo_new_path(cr);
            cairo_move_to(cr, f->x, f->y);
            cairo_line_to(cr, f->x - f->vx * 0.025, f->y - f->vy * 0.025);
            cairo_set_line_width(cr, 2.2);
            cairo_set_source_rgba(cr, 1, 0.75 * a + 0.2, 0.3 * a, a);
            cairo_stroke(cr);
            cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
        }
    }

    /* Tokens/sec over the sun */
    if (asleep) {
        neon_text(cr, c, 176, 38, 220, (rgb){ 0.6, 0.6, 0.75 }, NEON_CYAN, "0 TOK/S");
    } else {
        snprintf(txt, sizeof(txt), "%.0f", tok);
        neon_text(cr, c, 168, 80, 240, WHITE, NEON_PINK, txt);
        neon_text(cr, c, 218, 22, 120, NEON_CYAN, NEON_CYAN, "TOK/S");
    }

    /* Caption */
    if (caption_override) {
        int agi = !strncmp(caption_override, "AGI", 3);
        neon_text(cr, c, 64, 50, 330, agi ? (rgb){ 0.6, 1.0, 0.7 } : WHITE, agi ? (rgb){ 0.2, 1.0, 0.4 } : NEON_CYAN,
                  caption_override);
    } else if (asleep) {
        neon_text(cr, c, 64, 50, 300, WHITE, NEON_CYAN, "WEN PROMPT?");
    } else if (tok >= AGI_TOK && fmod(t, 9) >= 6) {
        neon_text(cr, c, 64, 50, 330, (rgb){ 0.6, 1.0, 0.7 }, (rgb){ 0.2, 1.0, 0.4 }, "AGI ACHIEVED INTERNALLY");
    } else if (tok >= HOT_TOK && fmod(t, 6) >= 3) {
        neon_text(cr, c, 64, 50, 300, (rgb){ 1.0, 0.62, 0.2 }, (rgb){ 1.0, 0.3, 0.1 }, "THIS IS FINE");
    } else {
        int rs = 2 + (int)fmin(12, tok / 150);
        char brr[32] = "BR";
        for (int i = 0; i < rs && i < 28; i++)
            strcat(brr, "R");
        snprintf(txt, sizeof(txt), "GPU GO %s", tok < 300 ? "BRR" : brr);
        neon_text(cr, c, 64, 50, 330, WHITE, NEON_PINK, txt);
    }

    /* Bottom: each server's tok/s either side of total watts */
    snprintf(txt, sizeof(txt), "%.0fW", total_w);
    neon_text(cr, c, 446, 30, 110, WHITE, NEON_PINK, txt);
    snprintf(txt, sizeof(txt), "%.0f", s->tok_port[0]);
    neon_text(cr, c - 104, 434, 28, 90, BLUE, BLUE, txt);
    snprintf(txt, sizeof(txt), "%.0f", s->tok_port[1]);
    neon_text(cr, c + 104, 434, 28, 90, ORANGE, ORANGE, txt);

    /* CRT scanlines and a round vignette */
    cairo_set_source_surface(cr, scanlines, 0, 0);
    cairo_paint(cr);
    {
        cairo_pattern_t *v = cairo_pattern_create_radial(c, c, 170, c, c, 250);
        cairo_pattern_add_color_stop_rgba(v, 0, 0, 0, 0, 0);
        cairo_pattern_add_color_stop_rgba(v, 1, 0, 0, 0, 0.75);
        cairo_set_source(cr, v);
        cairo_paint(cr);
        cairo_pattern_destroy(v);
    }
}

/* ---------------------------------------------------------------- showcase */

/*
 * A scripted 44 second loop that hits every mood in order, for filming the screen.
 * Each keyframe sets tok/s at its start time (eased toward the next) and an optional caption.
 */
typedef struct { double t, tok; const char *caption; } keyframe;

static const keyframe script[] = {
    {  0.0,    0, NULL },                       /* asleep: WEN PROMPT? */
    {  5.0,    0, NULL },
    {  5.4,   60, "PROMPT RECEIVED" },
    {  8.0,  300, NULL },                       /* BRRR builds */
    { 14.0,  950, NULL },                       /* shades drop at 800 */
    { 20.0, 1550, NULL },                       /* flames, THIS IS FINE */
    { 25.5, 2250, "AGI ACHIEVED INTERNALLY" },
    { 31.0, 2250, "KV CACHE FULL" },
    { 34.0,  350, "KV CACHE FULL" },
    { 36.5,    0, NULL },                       /* back to sleep */
    { 44.0,    0, NULL },
};
#define SCRIPT_LEN      (sizeof(script) / sizeof(script[0]))
#define SCRIPT_PERIOD   44.0

static void showcase_poll(stats *s, double t)
{
    double lt = fmod(t, SCRIPT_PERIOD), tok = 0;
    caption_override = NULL;
    for (size_t i = 0; i + 1 < SCRIPT_LEN; i++) {
        const keyframe *a = &script[i], *b = &script[i + 1];
        if (lt >= a->t && lt < b->t) {
            double u = (lt - a->t) / (b->t - a->t);
            u = u * u * (3 - 2 * u);                        /* smoothstep */
            tok = a->tok + (b->tok - a->tok) * u;
            caption_override = a->caption;
            break;
        }
    }
    double wobble = 1 + 0.04 * sin(t * 2.3);
    s->tok_port[0] = tok * 0.46 * wobble;
    s->tok_port[1] = tok * 0.54 / wobble;
    s->tok_s = s->tok_port[0] + s->tok_port[1];
    s->running = tok > 0 ? 1 + (int)(tok / 280) : 0;
    s->power[0] = 28 + fmin(547, tok * 0.26);
    s->power[1] = 33 + fmin(542, tok * 0.28);
    s->temp[0] = 36 + (int)(tok / 110);
    s->temp[1] = 42 + (int)(tok / 70);
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
    int fd = -1, bench = 0, showcase = 0;
    double fps = FPS;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--demo"))
            demo = 1;
        else if (!strcmp(argv[i], "--showcase"))
            showcase = demo = 1, fps = 30;
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
    init_scene();

    surf = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cr = cairo_create(surf);
    tj = tj3Init(TJINIT_COMPRESS);
    tj3Set(tj, TJPARAM_QUALITY, JPEG_QUALITY);
    tj3Set(tj, TJPARAM_SUBSAMP, TJSAMP_420);

    const double llama_x = WHEEL_CX - SPR_W * PX / 2.0 + 6;
    const double llama_y = WHEEL_CY + WHEEL_R - WHEEL_RIM - 23 * PX;

    if (bench) {
        struct { double w0, w1, tok; int run; double at; const char *png; } scenes[] = {
            { 470, 460, 1607, 8, 4.0, "brrr_preview.png" },    /* THIS IS FINE phase */
            { 470, 470, 2100, 8, 7.0, "brrr_agi.png" },        /* AGI ACHIEVED phase */
            { 330, 300, 900, 3, 1.0, "brrr_mid.png" },
            { 25, 30, 0, 0, 1.0, "brrr_idle.png" },
        };
        for (int k = 0; k < 4; k++) {
            memset(fxs, 0, sizeof(fxs));
            shades_t = -1;
            activity = 0;
            s.power[0] = scenes[k].w0; s.power[1] = scenes[k].w1;
            s.tok_s = scenes[k].tok; s.tok_port[0] = scenes[k].tok * 0.45; s.tok_port[1] = scenes[k].tok * 0.55;
            s.running = scenes[k].run;
            sh.tok = scenes[k].tok;
            double b0 = now_s();
            int n = 90;
            for (int i = 0; i < n; i++) {
                double tt = scenes[k].at - (n - 1 - i) / (double)FPS;
                simulate(&s, sh.tok, 1.0 / FPS, tt, llama_x, llama_y);
                render(cr, &s, &sh, tt);
                cairo_surface_flush(surf);
                size_t len = 0;
                tj3Compress8(tj, cairo_image_surface_get_data(surf), SIZE, cairo_image_surface_get_stride(surf),
                             SIZE, TJPF_BGRX, &jpeg, &len);
                if (i == n - 1)
                    printf("%s: %.2f ms/frame, jpeg %zu bytes\n", scenes[k].png, (now_s() - b0) * 1000 / n, len);
            }
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

        if (showcase) {
            showcase_poll(&s, t - t0);
        } else if (t >= next_poll) {
            next_poll = t + 1.0;
            if (demo) {
                demo_poll(&s, t - t0);
                /* demo: scale up so every mood shows up in one cycle */
                s.tok_port[0] *= 8.5;
                s.tok_port[1] *= 8.5;
                s.tok_s = s.tok_port[0] + s.tok_port[1];
                if (s.tok_s < 150) { s.tok_s = s.tok_port[0] = s.tok_port[1] = 0; s.running = 0; }
            } else {
                gpus_poll(&s);
                vllm_poll(&s, t);
            }
        }

        k = dt * 4 > 1 ? 1 : dt * 4;
        sh.zotac += (s.load[0] - sh.zotac) * k;
        sh.tuf   += (s.load[1] - sh.tuf) * k;
        sh.tok   = showcase ? s.tok_s : sh.tok + (s.tok_s - sh.tok) * k;

        simulate(&s, sh.tok, dt, t - t0, llama_x, llama_y);
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

        double spare = 1.0 / fps - (now_s() - t);
        if (spare > 0) {
            struct timespec ts = { 0, (long)(spare * 1e9) };
            nanosleep(&ts, NULL);
        }
    }

    tj3Free(jpeg);
    tj3Destroy(tj);
    cairo_surface_destroy(scanlines);
    cairo_destroy(cr);
    cairo_surface_destroy(surf);
    if (nvml_ok)
        nvmlShutdown();
    if (fd >= 0)
        close(fd);
    return 0;
}
