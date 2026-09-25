/*
 * autumn: a layered autumn valley at golden hour on the iCUE LINK AIO pump LCD, where
 * the falling maple leaves are your LLM's tokens.
 *
 * Dark branches frame the top: the golden one on the left drops GPU 0's vLLM server's
 * tokens, the red one on the right GPU 1's (one leaf per 16 tokens). The wind picks up
 * with total tokens/sec. Sky, sun, hills and branches are drawn once and cached, and
 * the frame rate drops to 15 fps when idle, so it's cheap to run.
 * Run with --demo to simulate data, --bench to write preview PNGs.
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
 * Everything that doesn't move (sky, sun, hills, foreground, branches) is drawn
 * once into two cached surfaces. Each frame only blits them and draws the falling
 * leaves and text, and the frame rate drops when idle, so CPU use stays low.
 */
#define FPS_BUSY        20
#define FPS_IDLE        15
#define TOKENS_PER_LEAF 16.0        /* one falling leaf per this many generated tokens */
#define MAX_LEAVES      500
#define HORIZON_Y       300.0
#define FG_Y            404.0       /* top of the dark foreground; leaves vanish into it */

typedef struct { double x, y; } pt;

/* Leaf clusters on the two framing branches: [0] left (GPU 0, gold), [1] right (GPU 1, red) */
#define N_ATTACH 22
static pt attach[2][N_ATTACH];

typedef struct {
    double x, y, vx, vy, rot, spin, flip, size, fade;
    int    side, shade, alive;
} leaf;

static leaf   leaves[MAX_LEAVES];
static double leaf_acc[3], wind, gust_t;
static cairo_surface_t *bg_cache, *fg_cache;

static double frand(void) { return rand() / (double)RAND_MAX; }

static rgb leaf_color(int side, int shade)
{
    static const rgb gold[4] = { { 0.99, 0.76, 0.22 }, { 0.96, 0.60, 0.14 }, { 1.00, 0.84, 0.36 }, { 0.90, 0.52, 0.12 } };
    static const rgb red[4] = { { 0.90, 0.26, 0.12 }, { 0.98, 0.42, 0.14 }, { 0.74, 0.16, 0.12 }, { 0.95, 0.33, 0.20 } };
    return side == 0 ? gold[shade & 3] : red[shade & 3];
}

/* Maple leaf outline, unit size, tip up */
static void maple_path(cairo_t *cr)
{
    static const double p[][2] = {
        { 0.00, -1.00 }, { 0.16, -0.58 }, { 0.58, -0.78 }, { 0.44, -0.34 }, { 0.98, -0.18 },
        { 0.62, 0.06 }, { 0.72, 0.42 }, { 0.20, 0.26 }, { 0.06, 0.52 },
        { -0.06, 0.52 }, { -0.20, 0.26 }, { -0.72, 0.42 }, { -0.62, 0.06 }, { -0.98, -0.18 },
        { -0.44, -0.34 }, { -0.58, -0.78 }, { -0.16, -0.58 },
    };
    cairo_new_path(cr);
    for (size_t i = 0; i < sizeof(p) / sizeof(p[0]); i++)
        cairo_line_to(cr, p[i][0], p[i][1]);
    cairo_close_path(cr);
}

static double ridge(double x, double base, double amp, double f, double ph)
{
    return base - amp * (0.6 * sin(x * f + ph) + 0.3 * sin(x * f * 2.3 + ph * 1.7) + 0.1 * sin(x * f * 5.1 + ph * 3.1));
}

/* One hill layer with tree silhouettes along its ridge */
static void hill_layer(cairo_t *cr, double base, double amp, double f, double ph, rgb col, int trees, double tree_h, int pines)
{
    cairo_new_path(cr);
    cairo_move_to(cr, 0, SIZE);
    for (double x = 0; x <= SIZE; x += 4)
        cairo_line_to(cr, x, ridge(x, base, amp, f, ph));
    cairo_line_to(cr, SIZE, SIZE);
    cairo_close_path(cr);
    set_rgb(cr, col);
    cairo_fill(cr);

    for (int i = 0; i < trees; i++) {
        double x = (i + 0.5 + (frand() - 0.5) * 0.8) * SIZE / trees;
        double y = ridge(x, base, amp, f, ph) + 2;
        double h = tree_h * (0.6 + frand() * 0.6);
        cairo_new_path(cr);
        if (pines) {
            for (int k = 0; k < 3; k++) {
                double yy = y - h * k / 3.2, w = h * (0.34 - k * 0.08);
                cairo_move_to(cr, x - w, yy);
                cairo_line_to(cr, x, yy - h * 0.5);
                cairo_line_to(cr, x + w, yy);
                cairo_close_path(cr);
            }
        } else {
            cairo_rectangle(cr, x - h * 0.05, y - h * 0.5, h * 0.1, h * 0.5);
            cairo_arc(cr, x, y - h * 0.62, h * 0.30, 0, 2 * M_PI);
            cairo_arc(cr, x - h * 0.18, y - h * 0.5, h * 0.2, 0, 2 * M_PI);
            cairo_arc(cr, x + h * 0.18, y - h * 0.52, h * 0.21, 0, 2 * M_PI);
        }
        set_rgb(cr, col);
        cairo_fill(cr);
    }
}

/* A dark framing branch from a top corner, with leaf clusters; fills attach[side] */
static void branch(cairo_t *cr, int side)
{
    const rgb BARK = { 0.11, 0.04, 0.07 };
    double dir = side == 0 ? 1 : -1, x0 = side == 0 ? -10 : SIZE + 10;
    int na = 0;

    /* main limb: a tapering curve */
    double px = x0, py = 40 + side * 14;
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    set_rgb(cr, BARK);
    for (int seg = 0; seg < 7; seg++) {
        double nx = px + dir * (26 + frand() * 6), ny = py + 7 + frand() * 8;
        cairo_move_to(cr, px, py);
        cairo_line_to(cr, nx, ny);
        cairo_set_line_width(cr, 14 - seg * 1.7);
        cairo_stroke(cr);
        /* twigs off the limb */
        if (seg > 0) {
            double tx = nx + dir * (10 + frand() * 18), ty = ny + (seg & 1 ? 1 : -1) * (14 + frand() * 12);
            cairo_move_to(cr, nx, ny);
            cairo_line_to(cr, tx, ty);
            cairo_set_line_width(cr, 3);
            cairo_stroke(cr);
            for (int k = 0; k < 3 && na < N_ATTACH; k++)
                attach[side][na++] = (pt){ tx + (frand() - 0.5) * 16, ty + (frand() - 0.5) * 12 };
        }
        px = nx;
        py = ny;
    }
    while (na < N_ATTACH)
        attach[side][na++] = (pt){ px + (frand() - 0.5) * 20, py + (frand() - 0.5) * 16 };

    /* leaves still hanging on the branch */
    for (int i = 0; i < N_ATTACH; i++)
        for (int k = 0; k < 2; k++) {
            rgb c = leaf_color(side, rand() & 3);
            cairo_save(cr);
            cairo_translate(cr, attach[side][i].x + (frand() - 0.5) * 10, attach[side][i].y + (frand() - 0.5) * 8);
            cairo_rotate(cr, M_PI + (frand() - 0.5) * 1.6);
            cairo_scale(cr, 9 + frand() * 4, 9 + frand() * 4);
            maple_path(cr);
            cairo_restore(cr);
            set_rgb(cr, c);
            cairo_fill(cr);
        }
}

static void build_caches(void)
{
    const double c = SIZE / 2.0;
    srand(11);

    /* Background: sky, sun, haze layers */
    bg_cache = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cairo_t *cr = cairo_create(bg_cache);
    cairo_pattern_t *g = cairo_pattern_create_linear(0, 0, 0, HORIZON_Y);
    cairo_pattern_add_color_stop_rgb(g, 0.00, 0.17, 0.10, 0.27);
    cairo_pattern_add_color_stop_rgb(g, 0.45, 0.55, 0.22, 0.34);
    cairo_pattern_add_color_stop_rgb(g, 0.80, 0.95, 0.52, 0.38);
    cairo_pattern_add_color_stop_rgb(g, 1.00, 1.00, 0.76, 0.52);
    cairo_set_source(cr, g);
    cairo_paint(cr);
    cairo_pattern_destroy(g);

    g = cairo_pattern_create_radial(c, 262, 40, c, 262, 200);
    cairo_pattern_add_color_stop_rgba(g, 0, 1, 0.85, 0.6, 0.55);
    cairo_pattern_add_color_stop_rgba(g, 1, 1, 0.6, 0.4, 0);
    cairo_set_source(cr, g);
    cairo_paint(cr);
    cairo_pattern_destroy(g);
    cairo_arc(cr, c, 262, 56, 0, 2 * M_PI);
    cairo_set_source_rgb(cr, 1.0, 0.90, 0.68);
    cairo_fill(cr);

    /* Hills, back to front: warm haze fading into deep shadow */
    hill_layer(cr, 284, 14, 0.010, 1.0, (rgb){ 0.93, 0.58, 0.46 }, 0, 0, 0);
    hill_layer(cr, 306, 16, 0.014, 2.3, (rgb){ 0.80, 0.42, 0.37 }, 22, 16, 1);
    hill_layer(cr, 330, 18, 0.011, 4.1, (rgb){ 0.62, 0.28, 0.29 }, 14, 30, 0);
    hill_layer(cr, 364, 20, 0.009, 0.4, (rgb){ 0.41, 0.16, 0.20 }, 10, 44, 0);
    cairo_destroy(cr);

    /* Foreground overlay: dark ground plus the framing branches, transparent elsewhere */
    fg_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, SIZE);
    cr = cairo_create(fg_cache);
    hill_layer(cr, FG_Y + 8, 12, 0.012, 5.2, (rgb){ 0.16, 0.06, 0.10 }, 0, 0, 0);
    /* grass tufts on the foreground ridge */
    cairo_set_source_rgb(cr, 0.16, 0.06, 0.10);
    cairo_set_line_width(cr, 1.6);
    for (double x = 10; x < SIZE; x += 5 + frand() * 6) {
        double y = ridge(x, FG_Y + 8, 12, 0.012, 5.2);
        cairo_move_to(cr, x, y + 2);
        cairo_line_to(cr, x + (frand() - 0.5) * 6, y - 5 - frand() * 9);
    }
    cairo_stroke(cr);
    branch(cr, 0);
    branch(cr, 1);
    /* soft round vignette */
    g = cairo_pattern_create_radial(c, c, 185, c, c, 250);
    cairo_pattern_add_color_stop_rgba(g, 0, 0.08, 0.02, 0.05, 0);
    cairo_pattern_add_color_stop_rgba(g, 1, 0.08, 0.02, 0.05, 0.7);
    cairo_set_source(cr, g);
    cairo_paint(cr);
    cairo_pattern_destroy(g);
    cairo_destroy(cr);

    srand((unsigned)time(NULL));
}

/* ---------------------------------------------------------------- simulation */

static void spawn_leaf(int side)
{
    for (int i = 0; i < MAX_LEAVES; i++) {
        leaf *l = &leaves[i];
        if (l->alive)
            continue;
        if (side < 2) {
            pt a = attach[side][rand() % N_ATTACH];
            l->x = a.x;
            l->y = a.y;
            l->side = side;
        } else {
            l->x = frand() * SIZE;                  /* idle stray drifting in from above */
            l->y = -12;
            l->side = rand() & 1;
        }
        l->shade = rand() & 3;
        l->vx = 0;
        l->vy = 20 + frand() * 16;
        l->rot = frand() * 2 * M_PI;
        l->spin = (frand() - 0.5) * 3;
        l->flip = frand() * 6;
        l->size = 6 + frand() * 3.5;
        l->fade = 1;
        l->alive = 1;
        return;
    }
}

static int simulate(const stats *s, double dt, double t)
{
    int idle = s->tok_s < 1 && s->running == 0, airborne = 0;

    if (t > gust_t)
        gust_t = t + 3 + frand() * 5;
    double gust = exp(-pow((gust_t - t - 1.2) / 0.8, 2));
    double want = 8 + 60 * clamp01(s->tok_s / 1600) + 45 * gust * clamp01(s->tok_s / 800 + 0.2);
    wind += (want - wind) * fmin(1, dt * 1.5);

    double rates[3] = { fmin(s->tok_port[0], 1500) / TOKENS_PER_LEAF, fmin(s->tok_port[1], 1500) / TOKENS_PER_LEAF,
                        idle ? 0.3 : 0 };
    for (int k = 0; k < 3; k++) {
        leaf_acc[k] += rates[k] * dt;
        while (leaf_acc[k] >= 1) {
            spawn_leaf(k);
            leaf_acc[k] -= 1;
        }
    }

    for (int i = 0; i < MAX_LEAVES; i++) {
        leaf *l = &leaves[i];
        if (!l->alive)
            continue;
        airborne++;
        l->flip += dt * (2.5 + l->size * 0.2);
        double sway = sin(l->flip * 0.9 + i) * 28;
        l->vx += ((wind + sway) - l->vx) * fmin(1, dt * 1.8);
        l->x += l->vx * dt;
        l->y += (l->vy + 9 * sin(l->flip * 1.7)) * dt;
        l->rot += l->spin * dt;
        if (l->x > SIZE + 20)
            l->x -= SIZE + 40;
        if (l->y > FG_Y - 6)
            l->fade -= dt * 2.5;                    /* settle into the grass */
        if (l->fade <= 0 || l->y > SIZE)
            l->alive = 0;
    }
    return airborne;
}

/* ---------------------------------------------------------------- render */

static void text_center(cairo_t *cr, double x, double y, double size, rgb c, const char *s)
{
    cairo_text_extents_t ext;
    cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, size);
    cairo_text_extents(cr, s, &ext);
    cairo_move_to(cr, x - ext.width / 2 - ext.x_bearing, y - ext.height / 2 - ext.y_bearing);
    set_rgb(cr, c);
    cairo_show_text(cr, s);
}

typedef struct { double zotac, tuf, tok; } shown_t;

static void render(cairo_t *cr, const stats *s, const shown_t *sh)
{
    const double c = SIZE / 2.0;
    char txt[64];

    cairo_set_source_surface(cr, bg_cache, 0, 0);
    cairo_paint(cr);

    for (int i = 0; i < MAX_LEAVES; i++) {
        const leaf *l = &leaves[i];
        if (!l->alive)
            continue;
        double flip = cos(l->flip);
        rgb col = leaf_color(l->side, l->shade);
        if (flip < 0)
            col = lerp(col, (rgb){ 0.35, 0.10, 0.08 }, 0.3);     /* underside */
        cairo_save(cr);
        cairo_translate(cr, l->x, l->y);
        cairo_rotate(cr, l->rot);
        cairo_scale(cr, l->size * (0.2 + 0.8 * fabs(flip)), l->size);
        maple_path(cr);
        cairo_restore(cr);
        cairo_set_source_rgba(cr, col.r, col.g, col.b, clamp01(l->fade));
        cairo_fill(cr);
    }

    cairo_set_source_surface(cr, fg_cache, 0, 0);
    cairo_paint(cr);

    if (s->tok_s < 0.5 && s->running == 0)
        snprintf(txt, sizeof(txt), "quiet");
    else
        snprintf(txt, sizeof(txt), "%.0f%s", shown_rate(sh->tok), rate_suffix());
    text_center(cr, c, 434, 24, (rgb){ 1.0, 0.90, 0.80 }, txt);
    snprintf(txt, sizeof(txt), "%.0f W", s->power[0] + s->power[1]);
    text_center(cr, c, 459, 14, (rgb){ 0.92, 0.78, 0.70 }, txt);
    snprintf(txt, sizeof(txt), "%d\xC2\xB0", s->temp[0]);
    text_center(cr, c - 52, 459, 14, leaf_color(0, 0), txt);
    snprintf(txt, sizeof(txt), "%d\xC2\xB0", s->temp[1]);
    text_center(cr, c + 52, 459, 14, leaf_color(1, 1), txt);
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
    int fd = -1, bench = 0, airborne = 0;

    const char *source = getenv("LLM_REACTOR_SOURCE");
    gpu_source = source && !strcasecmp(source, "gpu");
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gpu-load"))
            gpu_source = 1;
        else if (!strcmp(argv[i], "--demo"))
            demo = 1;
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
        struct { double tok0, tok1; double w0, w1; const char *png; } scenes[] = {
            { 723, 884, 470, 460, "autumn_preview.png" },
            { 180, 60, 250, 120, "autumn_mid.png" },
            { 0, 0, 25, 30, "autumn_idle.png" },
        };
        for (int k = 0; k < 3; k++) {
            memset(leaves, 0, sizeof(leaves));
            s.tok_port[0] = scenes[k].tok0; s.tok_port[1] = scenes[k].tok1;
            s.tok_s = s.tok_port[0] + s.tok_port[1]; s.running = s.tok_s > 0 ? 4 : 0;
            s.power[0] = scenes[k].w0; s.power[1] = scenes[k].w1; s.temp[0] = 54; s.temp[1] = 71;
            sh.tok = s.tok_s;
            int n = 15 * FPS_BUSY;
            size_t len = 0;
            double b0 = 0;
            for (int i = 0; i < n; i++) {
                if (i == n - 60)
                    b0 = now_s();
                airborne = simulate(&s, 1.0 / FPS_BUSY, i / (double)FPS_BUSY);
                render(cr, &s, &sh);
                cairo_surface_flush(surf);
                tj3Compress8(tj, cairo_image_surface_get_data(surf), SIZE, cairo_image_surface_get_stride(surf),
                             SIZE, TJPF_BGRX, &jpeg, &len);
            }
            printf("%s: %.2f ms/frame, jpeg %zu bytes, %d leaves in the air\n", scenes[k].png,
                   (now_s() - b0) * 1000 / 60, len, airborne);
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

        airborne = simulate(&s, dt, t - t0);
        render(cr, &s, &sh);
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

        /* Slow down when there's little to animate */
        int idle = s.tok_s < 1 && s.running == 0 && airborne < 4;
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
    cairo_surface_destroy(bg_cache);
    cairo_surface_destroy(fg_cache);
    if (nvml_ok)
        nvmlShutdown();
    if (fd >= 0)
        close(fd);
    return 0;
}
