/*
 * lantern: a lantern festival on a mountain lake at night, on the iCUE LINK AIO pump
 * LCD, where every sky lantern is a burst of your LLM's tokens.
 *
 * Paper lanterns are lit on the water and float up into the night: moon-blue ones from
 * the left shore for GPU 0's vLLM server, amber ones from the right shore for GPU 1's
 * (one lantern per 70 tokens). They rise faster the harder the servers work, drift off
 * into the distance with the breeze and are mirrored in the lake. When nothing is
 * running the lake goes still and only the odd stray lantern drifts by.
 * The painted lake is an image in assets/lantern/; the lanterns, their glow and
 * reflections are drawn with cairo from sprites cached at startup.
 * Run with --demo to simulate data, --showcase for a scripted 40 s festival,
 * --bench to write preview PNGs.
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
#define FPS             24
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

__attribute__((unused)) static const rgb BLUE   = { 61 / 255.0, 174 / 255.0, 233 / 255.0 };
__attribute__((unused)) static const rgb ORANGE = { 233 / 255.0, 120 / 255.0, 61 / 255.0 };
__attribute__((unused)) static const rgb WHITE  = { 240 / 255.0, 240 / 255.0, 245 / 255.0 };
__attribute__((unused)) static const rgb DIM    = { 110 / 255.0, 115 / 255.0, 125 / 255.0 };
__attribute__((unused)) static const rgb TRACK  = { 38 / 255.0, 40 / 255.0, 48 / 255.0 };
__attribute__((unused)) static const rgb BG     = { 6 / 255.0, 7 / 255.0, 12 / 255.0 };

typedef struct {
    double load[N_GPUS];        /* 0..1 */
    double power[N_GPUS];       /* W */
    int    temp[N_GPUS];        /* C */
    double tok_s;               /* all servers */
    double tok_port[2];         /* per server: [0] ZOTAC's vLLM, [1] TUF's vLLM */
    int    running;
    int    running_port[2];     /* running requests per server */
} stats;

static volatile sig_atomic_t stop;
static int demo;

/*
 * Data source. By default the display is driven by vLLM tokens/sec. With
 * LLM_REACTOR_SOURCE=gpu in the environment (or --gpu-load) it is driven by GPU
 * activity instead: each GPU's load is mapped onto the same range the display
 * expects (GPU_FULL_RATE "tok/s" at 100%), so it works for any GPU workload, and the
 * text shows GPU % instead of tok/s. Activity is half utilisation, half power draw
 * between GPU_IDLE_W and GPU_MAX_W: utilisation alone can sit at 100% while the card
 * is barely working, the watts show how hard it really is.
 */
#define GPU_FULL_RATE   900.0
#define GPU_IDLE_W      40.0        /* board power at idle */
#define GPU_MAX_W       575.0       /* board power limit */
static int gpu_source;

/* Numbers as shown on screen: tok/s, or GPU activity % in GPU mode */
__attribute__((unused)) static double shown_rate(double tok)        /* totals: average % */
{
    return gpu_source ? fmin(100, tok / (N_GPUS * GPU_FULL_RATE) * 100) : tok;
}
__attribute__((unused)) static double shown_gpu_rate(double tok)    /* one GPU */
{
    return gpu_source ? fmin(100, tok / GPU_FULL_RATE * 100) : tok;
}

/* Units to match: on their own, upper case, spelled out, and right after a number */
__attribute__((unused)) static const char *rate_unit(void) { return gpu_source ? "% GPU" : "tok/s"; }
__attribute__((unused)) static const char *rate_unit_uc(void) { return gpu_source ? "% GPU" : "TOK/S"; }
__attribute__((unused)) static const char *rate_unit_long(void) { return gpu_source ? "GPU LOAD %" : "TOKENS / SEC"; }
__attribute__((unused)) static const char *rate_suffix(void) { return gpu_source ? "% GPU" : " tok/s"; }

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

static void set_rgb(cairo_t *cr, rgb c) { cairo_set_source_rgb(cr, c.r, c.g, c.b); }

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
            s->running_port[i] = 0;
            s->tok_port[i] *= 0.5;
            s->tok_s += s->tok_port[i];
            continue;
        }
        tokens   = metric_sum(buf, "vllm:generation_tokens_total");
        s->running_port[i] = (int)metric_sum(buf, "vllm:num_requests_running");
        running += s->running_port[i];
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

/*
 * GPU mode: turn each GPU's activity (see GPU_FULL_RATE) into a token rate and a
 * running-request count, so the display's thresholds and idle detection just work.
 */
static void gpu_rate_poll(stats *s)
{
    s->tok_s = 0;
    s->running = 0;
    for (int i = 0; i < N_GPUS; i++) {
        double pw = clamp01((s->power[i] - GPU_IDLE_W) / (GPU_MAX_W - GPU_IDLE_W));
        double a = 0.5 * clamp01(s->load[i]) + 0.5 * pw;
        if (a < 0.03)                   /* no idle jitter */
            a = 0;
        s->tok_port[i] = a * GPU_FULL_RATE;
        s->running_port[i] = a > 0.05 ? (int)ceil(a * 4) : 0;
        s->tok_s += a * GPU_FULL_RATE;
        s->running += a > 0.05;
    }
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

/*
 * The painted lake never changes, so it's loaded once. Lanterns live in a tiny 3D
 * world (sideways X, height above the water h, depth z) and are projected with a
 * pinhole camera, so far lanterns are smaller, rise slower on screen and sit closer
 * to the far shore, and each one's reflection is mirrored about its own spot on the
 * water. Lantern bodies and glows are pre-rendered at a ladder of sizes; each frame
 * only blits sprites and redraws the text strip when a number changes.
 */
#define FPS_BUSY            20
#define FPS_IDLE            15
#define TOKENS_PER_LANTERN  70.0    /* one lantern per this many generated tokens */
#define MAX_RATE            10.0    /* lanterns/sec per server at most */
#define MAX_LANTERNS        420
#define HORIZON_Y           258.0   /* far shore line in the painting */
#define CAM_H               104.0   /* px below the horizon where a depth-1 lantern meets the water */
#define LANTERN_H           36.0    /* lantern height in px at depth 1 */
#define N_SPR               64      /* pre-rendered sprite sizes */
#define SPR_MIN             3.0
#define SPR_MAX             44.0
#define HUD_Y               366     /* cached text strip */
#define HUD_H               100
#define N_STARS             46
#define MOON_X              330.0   /* the painted moon; its path of light on the water is animated */

typedef struct {
    double X, h, z;                 /* world: sideways, height above the water, depth (1 = near) */
    double age, life, float_t, vrise, vx, rise, phase, bright;
    int    side, alive;
} lantern;

typedef struct { double x, y, r, ph, sp; } star;

static lantern lanterns[MAX_LANTERNS];
static star    stars[N_STARS];
static cairo_surface_t *bg_img, *hud_cache, *body_spr[2][N_SPR], *halo_spr[2][N_SPR], *refl_spr[2], *glow_spr[2];
static double  spr_h[N_SPR];
static double  spawn_acc[3], wind, wind_target, wind_t, busy_glow[2];
static char    hud_key[128];

static double frand(void) { return rand() / (double)RAND_MAX; }

/* Lantern paper colours: [side][0] top (deep), [1] bottom (lit by the flame), [2] halo */
static const rgb PAPER[2][3] = {
    { { 0.20, 0.42, 0.90 }, { 0.84, 0.95, 1.00 }, { 0.38, 0.66, 1.00 } },     /* GPU 0: moon blue */
    { { 0.88, 0.30, 0.08 }, { 1.00, 0.94, 0.62 }, { 1.00, 0.58, 0.18 } },     /* GPU 1: amber */
};

static cairo_surface_t *load_asset(const char *name)
{
    char exe[512], path[1024];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n > 0) {
        exe[n] = 0;
        char *slash = strrchr(exe, '/');
        if (slash)
            *slash = 0;
        snprintf(path, sizeof(path), "%s/assets/lantern/%s", exe, name);
        cairo_surface_t *s = cairo_image_surface_create_from_png(path);
        if (cairo_surface_status(s) == CAIRO_STATUS_SUCCESS)
            return s;
        cairo_surface_destroy(s);
    }
    snprintf(path, sizeof(path), "assets/lantern/%s", name);
    cairo_surface_t *s = cairo_image_surface_create_from_png(path);
    if (cairo_surface_status(s) != CAIRO_STATUS_SUCCESS) {
        fprintf(stderr, "lantern: can't load %s (looked next to the binary and in ./assets/lantern)\n", name);
        exit(1);
    }
    return s;
}

/* A sky lantern of height h centred on (0,0): a paper bag, wider at the top, lit from below */
static void lantern_shape(cairo_t *cr, double h, int side)
{
    const rgb *p = PAPER[side];
    double w = h * 0.66, top = -h * 0.5, bot = h * 0.5;

    cairo_new_path(cr);
    cairo_move_to(cr, -w * 0.50, top + h * 0.05);
    cairo_curve_to(cr, -w * 0.30, top - h * 0.03, w * 0.30, top - h * 0.03, w * 0.50, top + h * 0.05);
    cairo_curve_to(cr, w * 0.47, top + h * 0.45, w * 0.40, bot - h * 0.2, w * 0.36, bot);
    cairo_curve_to(cr, w * 0.15, bot + h * 0.04, -w * 0.15, bot + h * 0.04, -w * 0.36, bot);
    cairo_curve_to(cr, -w * 0.40, bot - h * 0.2, -w * 0.47, top + h * 0.45, -w * 0.50, top + h * 0.05);
    cairo_close_path(cr);

    cairo_pattern_t *g = cairo_pattern_create_linear(0, top, 0, bot);
    cairo_pattern_add_color_stop_rgb(g, 0.00, p[0].r * 0.8, p[0].g * 0.8, p[0].b * 0.8);
    cairo_pattern_add_color_stop_rgb(g, 0.45, (p[0].r + p[1].r) / 2, (p[0].g + p[1].g) / 2, (p[0].b + p[1].b) / 2);
    cairo_pattern_add_color_stop_rgb(g, 1.00, p[1].r, p[1].g, p[1].b);
    cairo_set_source(cr, g);
    cairo_fill_preserve(cr);
    cairo_pattern_destroy(g);

    /* flame shining through the paper */
    g = cairo_pattern_create_radial(0, bot - h * 0.1, 0, 0, bot - h * 0.1, h * 0.62);
    cairo_pattern_add_color_stop_rgba(g, 0, 1, 1, 0.92, 0.85);
    cairo_pattern_add_color_stop_rgba(g, 1, 1, 1, 0.92, 0);
    cairo_set_source(cr, g);
    cairo_fill_preserve(cr);
    cairo_pattern_destroy(g);

    /* paper ribs and rim */
    if (h >= 12) {
        cairo_save(cr);
        cairo_clip(cr);
        cairo_set_line_width(cr, fmax(0.6, h * 0.03));
        cairo_set_source_rgba(cr, p[0].r * 0.5, p[0].g * 0.5, p[0].b * 0.5, 0.35);
        for (int k = -1; k <= 1; k += 2) {
            cairo_move_to(cr, k * w * 0.19, top);
            cairo_line_to(cr, k * w * 0.14, bot);
        }
        cairo_move_to(cr, -w, top + h * 0.14);
        cairo_curve_to(cr, -w * 0.3, top + h * 0.07, w * 0.3, top + h * 0.07, w, top + h * 0.14);
        cairo_stroke(cr);
        cairo_restore(cr);
    }
    cairo_new_path(cr);
    cairo_move_to(cr, -w * 0.36, bot);
    cairo_curve_to(cr, -w * 0.15, bot + h * 0.04, w * 0.15, bot + h * 0.04, w * 0.36, bot);
    cairo_set_line_width(cr, fmax(0.7, h * 0.045));
    cairo_set_source_rgba(cr, p[0].r * 0.35, p[0].g * 0.3, p[0].b * 0.3, 0.9);
    cairo_stroke(cr);

    /* the flame itself, just under the opening */
    cairo_save(cr);
    cairo_translate(cr, 0, bot - h * 0.02);
    cairo_scale(cr, 1, 1.5);
    cairo_arc(cr, 0, 0, fmax(0.8, h * 0.06), 0, 2 * M_PI);
    cairo_restore(cr);
    cairo_set_source_rgb(cr, 1, 0.97, 0.80);
    cairo_fill(cr);
}

static cairo_surface_t *make_body(double h, int side)
{
    int w = (int)ceil(h * 0.8) + 4, hh = (int)ceil(h * 1.2) + 4;
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w, hh);
    cairo_t *cr = cairo_create(s);
    cairo_translate(cr, w / 2.0, hh / 2.0);
    lantern_shape(cr, h, side);
    cairo_destroy(cr);
    return s;
}

static cairo_surface_t *make_halo(double h, int side)
{
    double r = h * 1.9;
    int w = (int)ceil(r * 2) + 2;
    const rgb c = PAPER[side][2];
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w, w);
    cairo_t *cr = cairo_create(s);
    cairo_pattern_t *g = cairo_pattern_create_radial(w / 2.0, w / 2.0, 0, w / 2.0, w / 2.0, r);
    cairo_pattern_add_color_stop_rgba(g, 0.00, c.r, c.g, c.b, 0.55);
    cairo_pattern_add_color_stop_rgba(g, 0.18, c.r, c.g, c.b, 0.30);
    cairo_pattern_add_color_stop_rgba(g, 0.45, c.r, c.g, c.b, 0.09);
    cairo_pattern_add_color_stop_rgba(g, 1.00, c.r, c.g, c.b, 0);
    cairo_set_source(cr, g);
    cairo_paint(cr);
    cairo_pattern_destroy(g);
    cairo_destroy(cr);
    return s;
}

/* A soft vertical smear of light: a lantern's reflection on slightly rippled water */
static cairo_surface_t *make_refl(int side)
{
    const int w = 40, h = 120;
    const rgb c = lerp(PAPER[side][1], PAPER[side][2], 0.5);
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w, h);
    cairo_t *cr = cairo_create(s);
    cairo_translate(cr, w / 2.0, h / 2.0);
    cairo_scale(cr, 1, 3);
    cairo_pattern_t *g = cairo_pattern_create_radial(0, 0, 0, 0, 0, w / 2.0);
    cairo_pattern_add_color_stop_rgba(g, 0.0, c.r, c.g, c.b, 0.9);
    cairo_pattern_add_color_stop_rgba(g, 0.3, c.r, c.g, c.b, 0.35);
    cairo_pattern_add_color_stop_rgba(g, 1.0, c.r, c.g, c.b, 0);
    cairo_set_source(cr, g);
    cairo_paint(cr);
    cairo_pattern_destroy(g);
    cairo_destroy(cr);
    return s;
}

/* A wide, flat glow of lantern light spread over the water */
static cairo_surface_t *make_glow(int side)
{
    const int w = 300, h = 110;
    const rgb c = PAPER[side][2];
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w, h);
    cairo_t *cr = cairo_create(s);
    cairo_translate(cr, w / 2.0, h / 2.0);
    cairo_scale(cr, 1, h / (double)w);
    cairo_pattern_t *g = cairo_pattern_create_radial(0, 0, 0, 0, 0, w / 2.0);
    cairo_pattern_add_color_stop_rgba(g, 0.0, c.r, c.g, c.b, 0.35);
    cairo_pattern_add_color_stop_rgba(g, 0.5, c.r, c.g, c.b, 0.12);
    cairo_pattern_add_color_stop_rgba(g, 1.0, c.r, c.g, c.b, 0);
    cairo_set_source(cr, g);
    cairo_paint(cr);
    cairo_pattern_destroy(g);
    cairo_destroy(cr);
    return s;
}

static void build_caches(void)
{
    const double c = SIZE / 2.0;
    cairo_surface_t *src = load_asset("lake.png");
    bg_img = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cairo_t *cr = cairo_create(bg_img);
    cairo_scale(cr, SIZE / (double)cairo_image_surface_get_width(src), SIZE / (double)cairo_image_surface_get_height(src));
    cairo_set_source_surface(cr, src, 0, 0);
    cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_BEST);
    cairo_paint(cr);
    cairo_destroy(cr);
    cairo_surface_destroy(src);

    for (int i = 0; i < N_SPR; i++) {
        spr_h[i] = SPR_MIN * pow(SPR_MAX / SPR_MIN, i / (N_SPR - 1.0));
        for (int side = 0; side < 2; side++) {
            body_spr[side][i] = make_body(spr_h[i], side);
            halo_spr[side][i] = make_halo(spr_h[i], side);
        }
    }
    for (int side = 0; side < 2; side++) {
        refl_spr[side] = make_refl(side);
        glow_spr[side] = make_glow(side);
    }

    /* a few extra stars that twinkle over the painted ones */
    srand(7);
    for (int i = 0; i < N_STARS; i++) {
        double a, r;
        do {
            a = frand() * 2 * M_PI;
            r = sqrt(frand()) * 225;
            stars[i].x = c + cos(a) * r;
            stars[i].y = c + sin(a) * r;
        } while (stars[i].y > 150 || hypot(stars[i].x - MOON_X, stars[i].y - 78) < 30);
        stars[i].r = 0.7 + frand() * 1.1;
        stars[i].ph = frand() * 6.3;
        stars[i].sp = 0.6 + frand() * 1.8;
    }
    srand((unsigned)time(NULL));

    hud_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, HUD_H);
}

/* ---------------------------------------------------------------- simulation */

static void spawn_lantern(int side, double busy)
{
    for (int i = 0; i < MAX_LANTERNS; i++) {
        lantern *l = &lanterns[i];
        if (l->alive)
            continue;
        if (side > 1)
            side = rand() & 1;
        l->side = side;
        l->z = 1.1 + pow(frand(), 1.4) * 3.2;
        /* lit along its own shore: GPU 0 on the left, GPU 1 on the right */
        double base = HORIZON_Y + CAM_H / l->z, dy = base - SIZE / 2.0;
        double half = sqrt(fmax(0, 225.0 * 225.0 - dy * dy)) - 16;
        double sx = side == 0 ? SIZE / 2.0 - 8 - frand() * (half - 8) : SIZE / 2.0 + 8 + frand() * (half - 8);
        l->X = (sx - SIZE / 2.0) * l->z;
        l->h = LANTERN_H * 0.5;
        l->age = 0;
        l->life = 11 + frand() * 6;                 /* then it burns out */
        l->float_t = 0.7 + frand() * 1.4;
        l->vrise = (22 + 30 * busy) * (0.8 + frand() * 0.4);
        l->rise = 0;
        l->vx = (frand() - 0.5) * 40;
        l->phase = frand() * 20;
        l->bright = 0;
        l->alive = 1;
        return;
    }
}

/* Screen position, size and reflection of a lantern */
static void project(const lantern *l, double *sx, double *sy, double *hp, double *ry)
{
    double base = HORIZON_Y + CAM_H / l->z;
    *sx = SIZE / 2.0 + l->X / l->z;
    *sy = base - l->h / l->z;
    *ry = base + l->h / l->z;
    *hp = LANTERN_H / l->z;
}

static int simulate(const stats *s, double dt, double t)
{
    int idle = s->tok_s < 1 && s->running == 0, alive = 0;
    double busy = clamp01(s->tok_s / 1600);

    if (t > wind_t) {
        wind_t = t + 6 + frand() * 6;
        wind_target = -9 + frand() * 20;
    }
    wind += (wind_target - wind) * fmin(1, dt * 0.3);

    double rates[3] = { fmin(s->tok_port[0] / TOKENS_PER_LANTERN, MAX_RATE),
                        fmin(s->tok_port[1] / TOKENS_PER_LANTERN, MAX_RATE), idle ? 0.12 : 0 };
    for (int k = 0; k < 3; k++) {
        spawn_acc[k] += rates[k] * dt;
        while (spawn_acc[k] >= 1) {
            spawn_lantern(k, busy);
            spawn_acc[k] -= 1;
        }
        if (k < 2)
            busy_glow[k] += (clamp01(s->tok_port[k] / 700) - busy_glow[k]) * fmin(1, dt * 0.5);
    }

    for (int i = 0; i < MAX_LANTERNS; i++) {
        lantern *l = &lanterns[i];
        if (!l->alive)
            continue;
        alive++;
        l->age += dt;
        l->bright = fmin(1, l->bright + dt * 1.6);          /* being lit */
        if (l->age > l->float_t) {
            l->rise = fmin(1, l->rise + dt * 0.5);          /* lifts off gently */
            l->h += l->vrise * l->rise * l->rise * dt;
            l->z += l->z * 0.05 * l->rise * dt;             /* drifts away */
            l->X += (wind + l->vx + 7 * sin(l->age * 0.6 + l->phase)) * l->rise * dt;
        } else {
            l->X += 3 * sin(l->age * 1.3 + l->phase) * dt;   /* bobbing on the water */
        }
        double sx, sy, hp, ry;
        project(l, &sx, &sy, &hp, &ry);
        double d = hypot(sx - SIZE / 2.0, sy - SIZE / 2.0);
        if (l->age > l->life || (d > 246 && sy < SIZE / 2.0) || hp < SPR_MIN || sx < -60 || sx > SIZE + 60)
            l->alive = 0;
    }
    return alive;
}

/* ---------------------------------------------------------------- render */

static void blit(cairo_t *cr, cairo_surface_t *spr, double x, double y, double k, double alpha);

static void soft_text(cairo_t *cr, double x, double y, double size, rgb c, const char *s)
{
    cairo_text_extents_t ext;
    cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, size);
    cairo_text_extents(cr, s, &ext);
    cairo_move_to(cr, x - ext.width / 2 - ext.x_bearing, y - ext.height / 2 - ext.y_bearing);
    cairo_text_path(cr, s);
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
    cairo_set_line_width(cr, size * 0.24);
    cairo_set_source_rgba(cr, 0.01, 0.02, 0.07, 0.85);
    cairo_stroke_preserve(cr);
    set_rgb(cr, c);
    cairo_fill(cr);
}

typedef struct { double tok, tok0, tok1; } shown_t;

static void update_hud(const stats *s, const shown_t *sh, double t)
{
    const double c = SIZE / 2.0;
    static double tok, tok0, tok1, watts, next;
    char key[128], txt[64];
    if (t >= next || t < next - 1) {
        next = t + 0.25;
        tok = shown_rate(sh->tok);
        tok0 = shown_gpu_rate(sh->tok0);
        tok1 = shown_gpu_rate(sh->tok1);
        watts = s->power[0] + s->power[1];
    }
    int idle = s->tok_s < 1 && s->running == 0;
    snprintf(key, sizeof(key), "%.0f|%.0f|%.0f|%.0f|%d|%d|%d", tok, tok0, tok1, watts, s->temp[0], s->temp[1], idle);
    if (!strcmp(key, hud_key))
        return;
    strcpy(hud_key, key);

    cairo_t *cr = cairo_create(hud_cache);
    cairo_set_operator(cr, CAIRO_OPERATOR_CLEAR);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    cairo_translate(cr, 0, -HUD_Y);
    rgb blue = PAPER[0][2], amber = PAPER[1][2];
    blue = lerp(blue, (rgb){ 1, 1, 1 }, 0.25);
    amber = lerp(amber, (rgb){ 1, 1, 1 }, 0.15);
    if (idle) {
        soft_text(cr, c, 396, 32, (rgb){ 0.92, 0.94, 1.0 }, "still night");
        snprintf(txt, sizeof(txt), "%d\xC2\xB0", s->temp[0]);
        soft_text(cr, c - 88, 434, 22, blue, txt);
        snprintf(txt, sizeof(txt), "%.0f W", watts);
        soft_text(cr, c, 434, 22, (rgb){ 0.86, 0.88, 0.95 }, txt);
        snprintf(txt, sizeof(txt), "%d\xC2\xB0", s->temp[1]);
        soft_text(cr, c + 88, 434, 22, amber, txt);
    } else {
        snprintf(txt, sizeof(txt), "%.0f%s", tok, rate_suffix());
        soft_text(cr, c, 396, 36, (rgb){ 1.0, 0.97, 0.90 }, txt);
        snprintf(txt, sizeof(txt), "%.0f", tok0);
        soft_text(cr, c - 88, 434, 22, blue, txt);
        snprintf(txt, sizeof(txt), "%.0f W", watts);
        soft_text(cr, c, 434, 22, (rgb){ 0.86, 0.88, 0.95 }, txt);
        snprintf(txt, sizeof(txt), "%.0f", tok1);
        soft_text(cr, c + 88, 434, 22, amber, txt);
    }
    cairo_destroy(cr);
}

static int by_depth(const void *a, const void *b)
{
    double za = lanterns[*(const int *)a].z, zb = lanterns[*(const int *)b].z;
    return za < zb ? 1 : za > zb ? -1 : 0;
}

/* Blit sprite so that its centre lands on (x,y), scaled by k (<= 1) */
static void blit(cairo_t *cr, cairo_surface_t *spr, double x, double y, double k, double alpha)
{
    double w = cairo_image_surface_get_width(spr), h = cairo_image_surface_get_height(spr);
    if (k == 1) {                   /* fast path: translate only */
        cairo_set_source_surface(cr, spr, x - w / 2, y - h / 2);
        cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_BILINEAR);
        cairo_paint_with_alpha(cr, alpha);         /* EXTEND_NONE: only the sprite's area is touched */
        return;
    }
    cairo_save(cr);
    cairo_translate(cr, x, y);
    cairo_scale(cr, k, k);
    cairo_set_source_surface(cr, spr, -w / 2, -h / 2);
    cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_BILINEAR);
    cairo_rectangle(cr, -w / 2, -h / 2, w, h);
    cairo_clip(cr);
    cairo_paint_with_alpha(cr, alpha);
    cairo_restore(cr);
}

static void render(cairo_t *cr, const stats *s, const shown_t *sh, double t)
{
    static int order[MAX_LANTERNS];
    int n = 0;

    cairo_set_source_surface(cr, bg_img, 0, 0);
    cairo_paint(cr);

    /* twinkling stars */
    for (int i = 0; i < N_STARS; i++) {
        const star *st = &stars[i];
        double a = 0.5 + 0.5 * sin(t * st->sp + st->ph);
        cairo_arc(cr, st->x, st->y, st->r, 0, 2 * M_PI);
        cairo_set_source_rgba(cr, 0.9, 0.93, 1, a * a * 0.9);
        cairo_fill(cr);
    }

    /* moonlight shimmering on the water */
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    for (int i = 0; i < 26; i++) {
        double v = i / 25.0, y = HORIZON_Y + 24 + v * v * 170;
        double w = (3 + 16 * v) * (0.5 + 0.5 * sin(t * 0.7 + i * 1.9));
        double x = MOON_X + sin(t * 0.45 + i * 2.7) * (2 + 9 * v) + sin(i * 5.3) * 4 * v;
        double a = 0.5 + 0.5 * sin(t * (0.9 + 0.11 * i) + i * 4.1);
        cairo_move_to(cr, x - w / 2, y);
        cairo_line_to(cr, x + w / 2, y);
        cairo_set_line_width(cr, 1.0 + 1.0 * v);
        cairo_set_source_rgba(cr, 0.90, 0.90, 0.82, (0.06 + 0.34 * a * a) * (1 - 0.3 * v));
        cairo_stroke(cr);
    }

    /* the festival's glow on the water under each shore */
    for (int side = 0; side < 2; side++)
        if (busy_glow[side] > 0.01)
            blit(cr, glow_spr[side], side ? 340 : 140, HORIZON_Y + 30, 1, busy_glow[side] * 0.55);

    for (int i = 0; i < MAX_LANTERNS; i++)
        if (lanterns[i].alive)
            order[n++] = i;
    qsort(order, n, sizeof(int), by_depth);

    /* reflections first: on the water, under everything */
    cairo_save(cr);
    cairo_rectangle(cr, 0, HORIZON_Y + 1, SIZE, SIZE - HORIZON_Y);
    cairo_clip(cr);
    for (int j = 0; j < n; j++) {
        const lantern *l = &lanterns[order[j]];
        double sx, sy, hp, ry;
        project(l, &sx, &sy, &hp, &ry);
        if (ry - hp * 2 > SIZE)
            continue;
        double fl = 0.8 + 0.2 * sin(t * 9 + l->phase * 3) * sin(t * 5.3 + l->phase);
        double wob = sin(t * 1.7 + ry * 0.09 + l->phase) * hp * 0.10;
        double a = 0.5 * l->bright * clamp01((l->life - l->age) / 2.5) * fl * clamp01((hp - SPR_MIN) / 6);
        a *= 1 - 0.55 * clamp01((ry - HORIZON_Y - 2 * CAM_H / l->z) / 160);   /* high lanterns reflect fainter */
        double k = hp / 38.0;
        cairo_save(cr);
        cairo_translate(cr, sx + wob, ry);
        cairo_scale(cr, k * 0.85, k * 0.8);
        cairo_set_source_surface(cr, refl_spr[l->side], -20, -60);
        cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_BILINEAR);
        cairo_rectangle(cr, -20, -60, 40, 120);
        cairo_clip(cr);
        cairo_paint_with_alpha(cr, a);
        cairo_restore(cr);
        /* broken glints across the ripples */
        if (hp > 9) {
            const rgb c = PAPER[l->side][1];
            cairo_set_line_width(cr, fmax(1, hp * 0.06));
            for (int g = 0; g < 3; g++) {
                double gy = ry + (g - 1) * hp * 0.35 + hp * 0.1, gw = hp * (0.35 - 0.08 * abs(g - 1));
                double gx = sx + sin(t * 2.3 + g * 2.1 + l->phase) * hp * 0.12;
                cairo_move_to(cr, gx - gw / 2, gy);
                cairo_line_to(cr, gx + gw / 2, gy);
                cairo_set_source_rgba(cr, c.r, c.g, c.b, a * (0.6 + 0.4 * sin(t * 4 + g + l->phase)));
                cairo_stroke(cr);
            }
        }
    }
    cairo_restore(cr);

    /* lanterns, far to near */
    for (int j = 0; j < n; j++) {
        const lantern *l = &lanterns[order[j]];
        double sx, sy, hp, ry;
        project(l, &sx, &sy, &hp, &ry);
        double d = hypot(sx - SIZE / 2.0, sy - SIZE / 2.0);
        double edge = sy < SIZE / 2.0 ? clamp01((244 - d) / 40) : 1;
        double a = l->bright * clamp01((l->life - l->age) / 2.5) * edge * clamp01((hp - SPR_MIN) / 3) * (0.6 + 0.4 * clamp01((hp - 4) / 18));
        if (a <= 0.01)
            continue;
        double fl = 0.82 + 0.18 * sin(t * 8.3 + l->phase * 3) * sin(t * 4.9 + l->phase);
        /* nearest pre-rendered size, blitted unscaled (steps are 4%, too small to see) */
        int i = (int)lround(log(hp / SPR_MIN) / log(SPR_MAX / SPR_MIN) * (N_SPR - 1));
        i = i < 0 ? 0 : i >= N_SPR ? N_SPR - 1 : i;
        blit(cr, halo_spr[l->side][i], sx, sy + hp * 0.15, 1, a * fl);
        blit(cr, body_spr[l->side][i], sx, sy, 1, a);
    }

    update_hud(s, sh, t);
    cairo_set_source_surface(cr, hud_cache, 0, HUD_Y);
    cairo_paint(cr);
}

/* ---------------------------------------------------------------- showcase */

/*
 * --showcase: a scripted 40 s festival for filming or GIFs: a still night, GPU 0's
 * shore lights up alone, GPU 1 joins, both at full tilt, then the night goes quiet.
 */
static void showcase_poll(stats *s, double t)
{
    double u = fmod(t, 40.0);
    double a = 0, b = 0;
    if (u < 4)
        a = b = 0;
    else if (u < 12)
        a = 480 * clamp01((u - 4) / 1.5);
    else if (u < 19)
        a = 520, b = 420 * clamp01((u - 12) / 1.5);
    else if (u < 31)
        a = 820 + 60 * sin(u * 1.3), b = 780 + 50 * sin(u * 0.9 + 1);
    else if (u < 35)
        a = 0, b = 300 * (1 - clamp01((u - 31) / 3.5));
    a += a > 0 ? (frand() - 0.5) * 30 : 0;
    b += b > 0 ? (frand() - 0.5) * 30 : 0;
    s->tok_port[0] = a;
    s->tok_port[1] = b;
    s->tok_s = a + b;
    s->running_port[0] = a > 0 ? 1 + (int)(a / 200) : 0;
    s->running_port[1] = b > 0 ? 1 + (int)(b / 200) : 0;
    s->running = s->running_port[0] + s->running_port[1];
    s->load[0] = clamp01(a / 800);
    s->load[1] = clamp01(b / 800);
    s->power[0] = 28 + 420 * s->load[0];
    s->power[1] = 32 + 410 * s->load[1];
    s->temp[0] = (int)(36 + 22 * s->load[0]);
    s->temp[1] = (int)(40 + 28 * s->load[1]);
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
    int fd = -1, bench = 0, showcase = 0, alive = 0;

    const char *source = getenv("LLM_REACTOR_SOURCE");
    gpu_source = source && !strcasecmp(source, "gpu");
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gpu-load"))
            gpu_source = 1;
        else if (!strcmp(argv[i], "--demo"))
            demo = 1;
        else if (!strcmp(argv[i], "--showcase"))
            showcase = demo = 1;
        else if (!strcmp(argv[i], "--bench"))
            bench = 1;
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);
    build_caches();

    surf = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cr = cairo_create(surf);
    tj = tj3Init(TJINIT_COMPRESS);
    tj3Set(tj, TJPARAM_QUALITY, JPEG_QUALITY);
    tj3Set(tj, TJPARAM_SUBSAMP, TJSAMP_420);

    if (bench) {
        struct { double tok0, tok1, w0, w1; const char *png; } scenes[] = {
            { 860, 810, 470, 455, "lantern_preview.png" },
            { 320, 40, 260, 70, "lantern_mid.png" },
            { 0, 0, 25, 30, "lantern_idle.png" },
        };
        for (int k = 0; k < 3; k++) {
            srand(5 + k);                   /* repeatable previews */
            memset(lanterns, 0, sizeof(lanterns));
            wind = 0, wind_target = 3, wind_t = 1e9;   /* a light, steady breeze */
            hud_key[0] = 0;
            s.tok_port[0] = scenes[k].tok0; s.tok_port[1] = scenes[k].tok1;
            s.tok_s = s.tok_port[0] + s.tok_port[1]; s.running = s.tok_s > 0 ? 4 : 0;
            s.power[0] = scenes[k].w0; s.power[1] = scenes[k].w1; s.temp[0] = 54; s.temp[1] = 67;
            sh.tok = s.tok_s; sh.tok0 = s.tok_port[0]; sh.tok1 = s.tok_port[1];
            int n = (k == 2 ? 40 : 20) * FPS_BUSY;
            size_t len = 0;
            double b0 = 0;
            for (int i = 0; i < n; i++) {
                if (i == n - 60)
                    b0 = now_s();
                double t = i / (double)FPS_BUSY;
                alive = simulate(&s, 1.0 / FPS_BUSY, t);
                render(cr, &s, &sh, t);
                cairo_surface_flush(surf);
                tj3Compress8(tj, cairo_image_surface_get_data(surf), SIZE, cairo_image_surface_get_stride(surf),
                             SIZE, TJPF_BGRX, &jpeg, &len);
            }
            printf("%s: %.2f ms/frame, jpeg %zu bytes, %d lanterns\n", scenes[k].png,
                   (now_s() - b0) * 1000 / 60, len, alive);
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
                /* scale a copy each second; scaling s itself would compound */
                static stats demo_base;
                demo_poll(&demo_base, t - t0);
                s = demo_base;
                s.tok_port[0] *= 6; s.tok_port[1] *= 6;
                s.tok_s = s.tok_port[0] + s.tok_port[1];
                if (s.tok_s < 150) { s.tok_s = s.tok_port[0] = s.tok_port[1] = 0; s.running = 0; }
            } else {
                gpus_poll(&s);
                if (gpu_source)
                    gpu_rate_poll(&s);
                else
                    vllm_poll(&s, t);
            }
        }

        k = dt * 4 > 1 ? 1 : dt * 4;
        sh.tok += (s.tok_s - sh.tok) * k;
        sh.tok0 += (s.tok_port[0] - sh.tok0) * k;
        sh.tok1 += (s.tok_port[1] - sh.tok1) * k;

        alive = simulate(&s, dt, t - t0);
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

        /* Slow down when only a stray lantern or two is drifting */
        int idle = s.tok_s < 1 && s.running == 0 && alive < 6 && !showcase;
        double spare = 1.0 / (idle ? FPS_IDLE : FPS_BUSY) - (now_s() - t);
        if (spare > 0) {
            struct timespec ts = { 0, (long)(spare * 1e9) };
            nanosleep(&ts, NULL);
        }
    }

    tj3Free(jpeg);
    tj3Destroy(tj);
    cairo_destroy(cr);
    cairo_surface_destroy(surf);
    cairo_surface_destroy(bg_img);
    cairo_surface_destroy(hud_cache);
    for (int side = 0; side < 2; side++) {
        cairo_surface_destroy(refl_spr[side]);
        cairo_surface_destroy(glow_spr[side]);
        for (int i = 0; i < N_SPR; i++) {
            cairo_surface_destroy(body_spr[side][i]);
            cairo_surface_destroy(halo_spr[side][i]);
        }
    }
    if (nvml_ok)
        nvmlShutdown();
    if (fd >= 0)
        close(fd);
    return 0;
}
