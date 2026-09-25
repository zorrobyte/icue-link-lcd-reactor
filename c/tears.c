/*
 * tears: a close-up sequel to thirst on the iCUE LINK AIO pump LCD.
 *
 * A child's face on the left; on the right a GPU (fans for eyes) holds a glass to the
 * child's cheek with a robot arm. Tears are the token activity: they well up and run
 * into the glass at the real token rate, and the child gets progressively sadder the
 * more that is taken (an exponentially fading memory of recent tokens, ~60 s), from
 * content to inconsolable, recovering when idle. Full glasses get slurped dry.
 * The mL counter uses the same datacenter-water estimate as thirst (ML_PER_TOKEN).
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

#define FPS_BUSY        20
#define FPS_IDLE        8
#define ML_PER_TOKEN    (500.0 / (30 * 300))    /* same datacenter-water estimate as thirst.c */
#define TOKENS_PER_TEAR 26.0
#define MAX_TEARS_PER_S 8.0         /* beyond this, more flow shows as a heavier wet stream */
#define TEARS_PER_GLASS 24.0
#define SADNESS_TAU     60.0        /* seconds: how long load lingers in the child's mood */
#define SADNESS_FULL    40000.0     /* recent tokens at which the child is inconsolable */
#define MAX_TEARS       90

/* Child (left) */
#define HEAD_X          166.0
#define HEAD_Y          232.0
#define HEAD_RX         116.0
#define HEAD_RY         124.0
#define EYE_L_X         122.0
#define EYE_R_X         214.0
#define EYE_Y           236.0
#define JAW_X           262.0       /* where tears drip off the jaw */
#define JAW_Y           338.0

/* Robot (right) */
#define SHOULDER_X      356.0
#define SHOULDER_Y      372.0
#define ARM_A           66.0        /* upper arm */
#define ARM_B           64.0        /* forearm */
#define GLASS_UP_X      270.0       /* glass centre when held to the jaw */
#define GLASS_UP_Y      386.0
#define GLASS_DOWN_X    318.0       /* lowered, next to the robot */
#define GLASS_DOWN_Y    440.0

typedef struct {
    double x, y, vy, size;
    int    phase;                   /* 0 welling on the lid, 1 rolling down the cheek, 2 falling */
    double u, grow;
    int    alive;
} tear;

static tear   tears[MAX_TEARS];
static double tear_acc, glass, recent, sadness, harvested_ml, fan, wet, arm_p;
static double slurp_t = -1, blink_t = -1, next_blink = 2;
static cairo_surface_t *hud_cache, *bg_cache;
static int bg_level = -1;
static char hud_key[128];

static double frand(void) { return rand() / (double)RAND_MAX; }
static double smooth(double x) { x = clamp01(x); return x * x * (3 - 2 * x); }

static const rgb SKIN    = { 1.00, 0.84, 0.72 };
static const rgb SKIN_SH = { 0.90, 0.63, 0.52 };
static const rgb HAIR    = { 0.42, 0.24, 0.14 };
static const rgb HAIR_HI = { 0.62, 0.40, 0.24 };
static const rgb INK     = { 0.22, 0.12, 0.10 };
static const rgb TEAR    = { 0.62, 0.84, 1.00 };
static const rgb SWEATER = { 0.36, 0.62, 0.66 };

/* Tear path on the cheek: from the lower lid of the right eye, over the cheek to the jaw */
static void cheek_path(double u, double *x, double *y)
{
    static const double p[][2] = { { 226, 256 }, { 240, 280 }, { 252, 306 }, { 259, 324 }, { JAW_X, JAW_Y } };
    double f = clamp01(u) * 4;
    int i = (int)f;
    if (i >= 4) { *x = p[4][0]; *y = p[4][1]; return; }
    double t = f - i;
    *x = p[i][0] + (p[i + 1][0] - p[i][0]) * t;
    *y = p[i][1] + (p[i + 1][1] - p[i][1]) * t;
}

/* Where the glass is: lowered by the robot, or held up to the jaw */
static void glass_pos(double *x, double *y)
{
    double k = smooth(arm_p);
    *x = GLASS_DOWN_X + (GLASS_UP_X - GLASS_DOWN_X) * k;
    *y = GLASS_DOWN_Y + (GLASS_UP_Y - GLASS_DOWN_Y) * k;
}

/* ---------------------------------------------------------------- simulation */

static void simulate(const stats *s, double dt, double t)
{
    int idle = s->tok_s < 1 && s->running == 0;

    recent = recent * exp(-dt / SADNESS_TAU) + s->tok_s * dt;
    sadness += (clamp01(recent / SADNESS_FULL) - sadness) * fmin(1, dt * 0.8);
    harvested_ml += s->tok_s * dt * ML_PER_TOKEN;
    fan += dt * (0.5 + s->tok_s / 250);
    wet += ((idle ? 0 : clamp01(s->tok_s / 1200)) - wet) * fmin(1, dt * 0.7);

    /* The arm raises the glass while tears flow, lowers it to slurp or when idle */
    if (slurp_t < 0 && glass >= 1 && arm_p > 0.95)
        slurp_t = 0;
    double want = idle ? 0 : 1;
    if (slurp_t >= 0) {
        want = 0.25;
        if (arm_p < 0.3) {
            slurp_t += dt;
            glass = fmax(0, 1 - slurp_t / 1.6);
            if (slurp_t >= 1.6) {
                slurp_t = -1;
                glass = 0;
            }
        }
    }
    arm_p += (want - arm_p) * fmin(1, dt * 2.2);

    /* Blinks while the eyes are open */
    if (t >= next_blink && sadness < 0.8) {
        blink_t = 0;
        next_blink = t + 3 + frand() * 4;
    }
    if (blink_t >= 0 && (blink_t += dt) > 0.2)
        blink_t = -1;

    /* Tears well up on the lid at the token rate */
    tear_acc += (idle ? 0 : fmin(s->tok_s / TOKENS_PER_TEAR, MAX_TEARS_PER_S)) * dt;
    while (tear_acc >= 1) {
        for (int i = 0; i < MAX_TEARS; i++)
            if (!tears[i].alive) {
                tears[i] = (tear){ 0, 0, 0, 0, 0, 0, 0, 1 };
                tears[i].size = 4.5 + frand() * 2 + 2.5 * sadness;
                break;
            }
        tear_acc -= 1;
    }
    double gx, gy;
    glass_pos(&gx, &gy);
    for (int i = 0; i < MAX_TEARS; i++) {
        tear *d = &tears[i];
        if (!d->alive)
            continue;
        if (d->phase == 0) {                        /* swelling on the lower lid */
            d->grow += dt * (2.5 + 2 * sadness);
            cheek_path(0, &d->x, &d->y);
            if (d->grow >= 1)
                d->phase = 1;
        } else if (d->phase == 1) {                 /* rolling down the cheek, speeding up */
            d->u += dt * (0.55 + d->u * 1.4);
            cheek_path(d->u, &d->x, &d->y);
            if (d->u >= 1) {
                d->phase = 2;
                d->vy = 40;
            }
        } else {                                    /* falling from the jaw */
            d->vy += 700 * dt;
            d->y += d->vy * dt;
            d->x += 12 * dt;
            if (arm_p > 0.9 && slurp_t < 0 && d->y >= gy - 30 && fabs(d->x - gx) < 22) {
                d->alive = 0;
                glass = fmin(1, glass + 1 / TEARS_PER_GLASS);
            } else if (d->y > 470) {
                d->alive = 0;                       /* missed: onto the sweater */
            }
        }
    }
}

/* ---------------------------------------------------------------- render */

static void soft_text(cairo_t *cr, double x, double y, double size, rgb c, const char *s)
{
    cairo_text_extents_t ext;
    cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, size);
    cairo_text_extents(cr, s, &ext);
    cairo_move_to(cr, x - ext.width / 2 - ext.x_bearing, y - ext.height / 2 - ext.y_bearing);
    cairo_text_path(cr, s);
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
    cairo_set_line_width(cr, size * 0.22);
    cairo_set_source_rgba(cr, 0.12, 0.10, 0.18, 0.75);
    cairo_stroke_preserve(cr);
    set_rgb(cr, c);
    cairo_fill(cr);
}

typedef struct { double zotac, tuf, tok; } shown_t;

static void radial_blob(cairo_t *cr, double x, double y, double r, rgb c, double a)
{
    cairo_pattern_t *g = cairo_pattern_create_radial(x, y, 0, x, y, r);
    cairo_pattern_add_color_stop_rgba(g, 0, c.r, c.g, c.b, a);
    cairo_pattern_add_color_stop_rgba(g, 1, c.r, c.g, c.b, 0);
    cairo_arc(cr, x, y, r, 0, 2 * M_PI);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
}

/* A big glossy storybook eye. open: 1 wide .. 0 shut. look: offset toward the glass */
static void draw_eye(cairo_t *cr, double cx, double cy, double open, double sad, int right)
{
    const double w = 30, h = 25;
    double top = cy - h * open + 6 * sad;           /* sad lids droop */
    double bot = cy + h * 0.85;

    /* the eye opening (almond), clipped */
    cairo_save(cr);
    cairo_new_path(cr);
    cairo_move_to(cr, cx - w, cy + 2);
    cairo_curve_to(cr, cx - w * 0.6, top - 6, cx + w * 0.6, top - 6, cx + w, cy + 2);
    cairo_curve_to(cr, cx + w * 0.6, bot + 4, cx - w * 0.6, bot + 4, cx - w, cy + 2);
    cairo_close_path(cr);
    cairo_set_source_rgb(cr, 1, 1, 1);
    cairo_fill_preserve(cr);
    cairo_clip(cr);

    /* big iris looking down-right toward the glass */
    double ix = cx + 6 + (right ? 2 : 0), iy = cy + 6;
    cairo_pattern_t *g = cairo_pattern_create_radial(ix, iy, 4, ix, iy, 22);
    cairo_pattern_add_color_stop_rgb(g, 0, 0.10, 0.06, 0.04);
    cairo_pattern_add_color_stop_rgb(g, 0.45, 0.30, 0.18, 0.10);
    cairo_pattern_add_color_stop_rgb(g, 0.85, 0.52, 0.34, 0.18);
    cairo_pattern_add_color_stop_rgb(g, 1, 0.22, 0.12, 0.06);
    cairo_arc(cr, ix, iy, 22, 0, 2 * M_PI);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    cairo_arc(cr, ix, iy, 10, 0, 2 * M_PI);
    cairo_set_source_rgb(cr, 0.04, 0.02, 0.02);
    cairo_fill(cr);
    /* catchlights, bigger and wetter when sad */
    cairo_arc(cr, ix - 8, iy - 9, 6.5 + 2 * sad, 0, 2 * M_PI);
    cairo_arc(cr, ix + 7, iy + 6, 3 + sad, 0, 2 * M_PI);
    cairo_set_source_rgba(cr, 1, 1, 1, 0.95);
    cairo_fill(cr);
    /* watery line along the lower lid */
    if (sad > 0.15) {
        cairo_save(cr);
        cairo_translate(cr, cx, bot - 2);
        cairo_scale(cr, 1, 0.25);
        cairo_arc(cr, 0, 0, w * 0.9, 0, 2 * M_PI);
        cairo_restore(cr);
        cairo_set_source_rgba(cr, TEAR.r, TEAR.g, TEAR.b, 0.3 + 0.4 * sad);
        cairo_fill(cr);
    }
    /* soft shadow from the upper lid */
    g = cairo_pattern_create_linear(0, top - 8, 0, top + 14);
    cairo_pattern_add_color_stop_rgba(g, 0, 0.45, 0.25, 0.20, 0.45);
    cairo_pattern_add_color_stop_rgba(g, 1, 0.45, 0.25, 0.20, 0);
    cairo_rectangle(cr, cx - w, top - 10, 2 * w, 26);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    cairo_restore(cr);

    /* upper lid line and lashes */
    cairo_new_path(cr);
    cairo_move_to(cr, cx - w - 2, cy + 3);
    cairo_curve_to(cr, cx - w * 0.6, top - 7, cx + w * 0.6, top - 7, cx + w + 3, cy + 1);
    cairo_set_line_width(cr, 4.5);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    set_rgb(cr, INK);
    cairo_stroke(cr);
    double ox = right ? cx + w : cx - w, dir = right ? 1 : -1;
    for (int k = 0; k < 3; k++) {
        cairo_move_to(cr, ox - dir * k * 7, cy - 2 - k * 4);
        cairo_line_to(cr, ox + dir * (7 - k * 2), cy - 9 - k * 6);
    }
    cairo_set_line_width(cr, 3);
    cairo_stroke(cr);
    /* lower lid, soft */
    cairo_move_to(cr, cx - w * 0.8, bot - 2);
    cairo_curve_to(cr, cx - w * 0.3, bot + 4, cx + w * 0.3, bot + 4, cx + w * 0.8, bot - 2);
    cairo_set_line_width(cr, 1.8);
    cairo_set_source_rgba(cr, INK.r, INK.g, INK.b, 0.5);
    cairo_stroke(cr);
}

static void draw_child(cairo_t *cr, double t)
{
    double sad = sadness;
    double sob = sad > 0.75 ? sin(t * 26) * 2.2 * (sad - 0.75) / 0.25 : 0;
    cairo_save(cr);
    cairo_translate(cr, 0, fabs(sob));

    /* sweater and neck */
    cairo_new_path(cr);
    cairo_move_to(cr, 10, SIZE + 10);
    cairo_curve_to(cr, 20, 390, 110, 360, HEAD_X, 362);
    cairo_curve_to(cr, 230, 360, 300, 390, 320, SIZE + 10);
    cairo_close_path(cr);
    cairo_pattern_t *g = cairo_pattern_create_linear(0, 360, 0, SIZE);
    cairo_pattern_add_color_stop_rgb(g, 0, SWEATER.r * 1.1, SWEATER.g * 1.1, SWEATER.b * 1.1);
    cairo_pattern_add_color_stop_rgb(g, 1, SWEATER.r * 0.7, SWEATER.g * 0.7, SWEATER.b * 0.7);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    cairo_set_line_width(cr, 2);
    cairo_set_source_rgba(cr, 1, 1, 1, 0.18);
    for (int k = 0; k < 7; k++) {                   /* knit ribs */
        double x = 60 + k * 36;
        cairo_move_to(cr, x, 404);
        cairo_line_to(cr, x + 4, SIZE);
    }
    cairo_stroke(cr);
    /* wet spots on the sweater from missed tears */
    if (wet > 0.3 && sad > 0.3)
        radial_blob(cr, 272, 440, 26 * wet, (rgb){ 0.20, 0.40, 0.45 }, 0.5 * sad);

    /* ear, behind the head on the far side */
    cairo_save(cr);
    cairo_translate(cr, 58, 250);
    cairo_scale(cr, 0.7, 1);
    cairo_arc(cr, 0, 0, 24, 0, 2 * M_PI);
    cairo_restore(cr);
    set_rgb(cr, SKIN_SH);
    cairo_fill(cr);

    /* head: soft egg with painterly shading */
    cairo_save(cr);
    cairo_translate(cr, HEAD_X, HEAD_Y);
    cairo_scale(cr, HEAD_RX / HEAD_RY, 1);
    cairo_new_path(cr);
    cairo_arc(cr, 0, 0, HEAD_RY, 0, 2 * M_PI);
    cairo_restore(cr);
    g = cairo_pattern_create_radial(HEAD_X + 30, HEAD_Y - 30, 20, HEAD_X, HEAD_Y + 10, HEAD_RY * 1.05);
    cairo_pattern_add_color_stop_rgb(g, 0, 1.0, 0.90, 0.80);
    cairo_pattern_add_color_stop_rgb(g, 0.7, SKIN.r, SKIN.g, SKIN.b);
    cairo_pattern_add_color_stop_rgb(g, 1, SKIN_SH.r, SKIN_SH.g, SKIN_SH.b);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);

    /* rosy cheeks, redder with crying */
    radial_blob(cr, 104, 292, 34, (rgb){ 1, 0.45, 0.45 }, 0.30 + 0.30 * sad);
    radial_blob(cr, 238, 290, 30, (rgb){ 1, 0.45, 0.45 }, 0.30 + 0.30 * sad);

    /* hair: a soft rounded mop with bangs and highlights */
    cairo_new_path(cr);
    cairo_move_to(cr, 44, 262);
    cairo_curve_to(cr, 26, 150, 100, 88, 172, 92);
    cairo_curve_to(cr, 250, 94, 300, 150, 286, 236);
    cairo_curve_to(cr, 270, 190, 250, 170, 232, 168);  /* right temple */
    cairo_curve_to(cr, 222, 186, 200, 192, 186, 176);   /* bang */
    cairo_curve_to(cr, 170, 196, 140, 196, 128, 178);   /* bang */
    cairo_curve_to(cr, 112, 196, 84, 196, 74, 184);     /* bang */
    cairo_curve_to(cr, 62, 206, 56, 236, 44, 262);
    cairo_close_path(cr);
    g = cairo_pattern_create_linear(0, 90, 0, 250);
    cairo_pattern_add_color_stop_rgb(g, 0, HAIR_HI.r, HAIR_HI.g, HAIR_HI.b);
    cairo_pattern_add_color_stop_rgb(g, 1, HAIR.r, HAIR.g, HAIR.b);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    cairo_set_line_width(cr, 3);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    cairo_set_source_rgba(cr, 1, 0.85, 0.65, 0.35);
    cairo_move_to(cr, 110, 118); cairo_curve_to(cr, 140, 104, 180, 104, 210, 116);
    cairo_move_to(cr, 96, 142); cairo_curve_to(cr, 120, 128, 150, 126, 170, 132);
    cairo_stroke(cr);

    /* brows: soft, inner ends lifting with sadness */
    double lift = 20 * sad;
    cairo_set_line_width(cr, 7);
    set_rgb(cr, HAIR);
    cairo_move_to(cr, EYE_L_X - 26, 196 + lift * 0.1);
    cairo_curve_to(cr, EYE_L_X - 10, 190 - lift * 0.3, EYE_L_X + 8, 190 - lift * 0.8, EYE_L_X + 22, 192 - lift);
    cairo_move_to(cr, EYE_R_X - 22, 192 - lift);
    cairo_curve_to(cr, EYE_R_X - 8, 190 - lift * 0.8, EYE_R_X + 10, 190 - lift * 0.3, EYE_R_X + 26, 196 + lift * 0.1);
    cairo_stroke(cr);

    /* eyes: open, drooping, blinking, then squeezed shut */
    if (sad < 0.82) {
        double open = 1 - 0.35 * sad;
        if (blink_t >= 0)
            open *= fabs(1 - blink_t / 0.1);
        draw_eye(cr, EYE_L_X, EYE_Y, fmax(0.05, open), sad, 0);
        draw_eye(cr, EYE_R_X, EYE_Y, fmax(0.05, open), sad, 1);
    } else {
        /* squeezed shut: > <, with rivers pouring down both cheeks */
        double k = (sad - 0.82) / 0.18;
        cairo_set_line_width(cr, 5.5);
        cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
        set_rgb(cr, INK);
        cairo_move_to(cr, EYE_L_X - 22, EYE_Y - 12);
        cairo_curve_to(cr, EYE_L_X - 4, EYE_Y - 6, EYE_L_X + 8, EYE_Y - 2, EYE_L_X + 18, EYE_Y + 2);
        cairo_curve_to(cr, EYE_L_X + 8, EYE_Y + 6, EYE_L_X - 4, EYE_Y + 10, EYE_L_X - 22, EYE_Y + 14);
        cairo_move_to(cr, EYE_R_X + 22, EYE_Y - 12);
        cairo_curve_to(cr, EYE_R_X + 4, EYE_Y - 6, EYE_R_X - 8, EYE_Y - 2, EYE_R_X - 18, EYE_Y + 2);
        cairo_curve_to(cr, EYE_R_X - 8, EYE_Y + 6, EYE_R_X + 4, EYE_Y + 10, EYE_R_X + 22, EYE_Y + 14);
        cairo_stroke(cr);
        for (int e = 0; e < 2; e++) {
            double ex = e ? EYE_R_X - 2 : EYE_L_X - 6, wob = sin(t * 8 + e) * 2;
            cairo_new_path(cr);
            cairo_move_to(cr, ex - 8, EYE_Y + 14);
            cairo_curve_to(cr, ex - 14 + wob, EYE_Y + 50, ex - 6, EYE_Y + 80, ex - 12 - wob, EYE_Y + 112);
            cairo_line_to(cr, ex + 6 + wob, EYE_Y + 112);
            cairo_curve_to(cr, ex + 12, EYE_Y + 80, ex + 4 - wob, EYE_Y + 50, ex + 8, EYE_Y + 14);
            cairo_close_path(cr);
            cairo_set_source_rgba(cr, TEAR.r, TEAR.g, TEAR.b, 0.55 + 0.25 * k);
            cairo_fill_preserve(cr);
            cairo_set_line_width(cr, 1.5);
            cairo_set_source_rgba(cr, 1, 1, 1, 0.6);
            cairo_stroke(cr);
            set_rgb(cr, INK);
            cairo_set_line_width(cr, 5.5);
        }
    }

    /* button nose with a soft shadow */
    radial_blob(cr, 186, 282, 16, SKIN_SH, 0.6);
    cairo_move_to(cr, 180, 288);
    cairo_curve_to(cr, 186, 292, 194, 290, 196, 284);
    cairo_set_line_width(cr, 2.5);
    cairo_set_source_rgba(cr, INK.r, INK.g, INK.b, 0.6);
    cairo_stroke(cr);

    /* mouth: smile, worried wobble, pout, wail */
    double mx = 176, my = 322;
    set_rgb(cr, INK);
    cairo_set_line_width(cr, 4);
    if (sad < 0.3) {
        cairo_move_to(cr, mx - 20, my - 2);
        cairo_curve_to(cr, mx - 8, my + 10 - sad * 20, mx + 8, my + 10 - sad * 20, mx + 20, my - 2);
        cairo_stroke(cr);
    } else if (sad < 0.6) {
        double wob = sin(t * 14) * 2 * (sad - 0.3) / 0.3;
        cairo_move_to(cr, mx - 20, my + 4);
        cairo_curve_to(cr, mx - 8, my - 6 + wob, mx + 8, my - 6 - wob, mx + 20, my + 4);
        cairo_stroke(cr);
    } else if (sad < 0.82) {
        /* trembling pout */
        double wob = sin(t * 22) * 1.5;
        cairo_move_to(cr, mx - 18, my + 6);
        cairo_curve_to(cr, mx - 6, my - 8 + wob, mx + 6, my - 8 - wob, mx + 18, my + 6);
        cairo_curve_to(cr, mx + 6, my + 2, mx - 6, my + 2, mx - 18, my + 6);
        cairo_set_source_rgb(cr, 0.75, 0.30, 0.32);
        cairo_fill_preserve(cr);
        set_rgb(cr, INK);
        cairo_stroke(cr);
    } else {
        /* wailing */
        double k = (sad - 0.82) / 0.18, w = 26 + 10 * k, h = 30 + 16 * k, wob = sin(t * 30) * 2;
        cairo_new_path(cr);
        cairo_move_to(cr, mx - w, my + 4);
        cairo_curve_to(cr, mx - w * 0.5, my - h * 0.4 + wob, mx + w * 0.5, my - h * 0.4 - wob, mx + w, my + 4);
        cairo_curve_to(cr, mx + w * 0.8, my + h, mx - w * 0.8, my + h, mx - w, my + 4);
        cairo_close_path(cr);
        cairo_set_source_rgb(cr, 0.40, 0.10, 0.14);
        cairo_fill_preserve(cr);
        set_rgb(cr, INK);
        cairo_stroke(cr);
        radial_blob(cr, mx, my + h * 0.62, w * 0.55, (rgb){ 0.95, 0.45, 0.50 }, 0.9);
    }

    /* glossy wet trail down the cheek */
    if (wet > 0.02) {
        cairo_new_path(cr);
        for (int k = 0; k <= 16; k++) {
            double x, y;
            cheek_path(k / 16.0, &x, &y);
            cairo_line_to(cr, x, y);
        }
        cairo_set_line_width(cr, 6 + 8 * wet);
        cairo_set_source_rgba(cr, TEAR.r, TEAR.g, TEAR.b, 0.25 + 0.35 * wet);
        cairo_stroke_preserve(cr);
        cairo_set_line_width(cr, 2 + wet);
        cairo_set_source_rgba(cr, 1, 1, 1, 0.3 + 0.3 * wet);
        cairo_stroke(cr);
    }
    /* the other eye's tears when crying hard */
    if (sad > 0.55) {
        double a = (sad - 0.55) / 0.45;
        cairo_new_path(cr);
        cairo_move_to(cr, EYE_L_X - 8, EYE_Y + 22);
        cairo_curve_to(cr, EYE_L_X - 16, EYE_Y + 50, EYE_L_X - 10, EYE_Y + 80, EYE_L_X - 18, EYE_Y + 104);
        cairo_set_line_width(cr, 7);
        cairo_set_source_rgba(cr, TEAR.r, TEAR.g, TEAR.b, 0.45 * a);
        cairo_stroke(cr);
    }
    cairo_restore(cr);
}

static void draw_tear(cairo_t *cr, const tear *d)
{
    double r = d->size * (d->phase == 0 ? 0.4 + 0.6 * d->grow : 1);
    double stretch = d->phase == 2 ? 1.25 : 1.05;
    cairo_save(cr);
    cairo_translate(cr, d->x, d->y);
    cairo_move_to(cr, 0, -r * 1.7 * stretch);
    cairo_curve_to(cr, r, -r * 0.3, r * 1.05, r, 0, r);
    cairo_curve_to(cr, -r * 1.05, r, -r, -r * 0.3, 0, -r * 1.7 * stretch);
    cairo_pattern_t *g = cairo_pattern_create_radial(-r * 0.3, -r * 0.3, 0, 0, 0, r * 1.4);
    cairo_pattern_add_color_stop_rgba(g, 0, 1, 1, 1, 0.95);
    cairo_pattern_add_color_stop_rgba(g, 0.35, TEAR.r, TEAR.g, TEAR.b, 0.85);
    cairo_pattern_add_color_stop_rgba(g, 1, 0.30, 0.55, 0.90, 0.9);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    cairo_restore(cr);
}

/* Two-link arm from the shoulder to a hand position (elbow bends downward) */
static void arm_to(cairo_t *cr, double hx, double hy)
{
    double dx = hx - SHOULDER_X, dy = hy - SHOULDER_Y, d = fmin(hypot(dx, dy), ARM_A + ARM_B - 1);
    double base = atan2(dy, dx);
    double cos_a = (ARM_A * ARM_A + d * d - ARM_B * ARM_B) / (2 * ARM_A * d);
    double a = acos(fmax(-1, fmin(1, cos_a)));
    /* reaching left, subtracting the angle puts the elbow below the line to the hand */
    double ex = SHOULDER_X + ARM_A * cos(base - a), ey = SHOULDER_Y + ARM_A * sin(base - a);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
    cairo_move_to(cr, SHOULDER_X, SHOULDER_Y);
    cairo_line_to(cr, ex, ey);
    cairo_line_to(cr, hx, hy);
    cairo_set_line_width(cr, 15);
    cairo_set_source_rgb(cr, 0.14, 0.14, 0.18);
    cairo_stroke_preserve(cr);
    cairo_set_line_width(cr, 9);
    cairo_set_source_rgb(cr, 0.62, 0.65, 0.74);
    cairo_stroke(cr);
    for (int j = 0; j < 2; j++) {
        double jx = j ? ex : SHOULDER_X, jy = j ? ey : SHOULDER_Y;
        cairo_arc(cr, jx, jy, 8, 0, 2 * M_PI);
        cairo_set_source_rgb(cr, 0.30, 0.31, 0.38);
        cairo_fill(cr);
    }
}

static void draw_glass(cairo_t *cr, double gx, double gy)
{
    const double wt = 46, wb = 36, h = 62;
    double y0 = gy - h / 2, y1 = gy + h / 2;
    /* liquid */
    if (glass > 0.01) {
        double lh = (h - 8) * glass, yt = y1 - 4 - lh, w = wb + (wt - wb) * ((lh + 4) / h);
        cairo_move_to(cr, gx - w / 2 + 3, yt);
        cairo_line_to(cr, gx + w / 2 - 3, yt);
        cairo_line_to(cr, gx + wb / 2 - 3, y1 - 4);
        cairo_line_to(cr, gx - wb / 2 + 3, y1 - 4);
        cairo_close_path(cr);
        cairo_pattern_t *g = cairo_pattern_create_linear(0, yt, 0, y1);
        cairo_pattern_add_color_stop_rgba(g, 0, 0.75, 0.90, 1.0, 0.9);
        cairo_pattern_add_color_stop_rgba(g, 1, 0.35, 0.60, 0.95, 0.9);
        cairo_set_source(cr, g);
        cairo_fill(cr);
        cairo_pattern_destroy(g);
        cairo_move_to(cr, gx - w / 2 + 4, yt);
        cairo_line_to(cr, gx + w / 2 - 4, yt);
        cairo_set_line_width(cr, 2);
        cairo_set_source_rgba(cr, 1, 1, 1, 0.8);
        cairo_stroke(cr);
    }
    /* the glass itself */
    cairo_move_to(cr, gx - wt / 2, y0);
    cairo_line_to(cr, gx + wt / 2, y0);
    cairo_line_to(cr, gx + wb / 2, y1);
    cairo_line_to(cr, gx - wb / 2, y1);
    cairo_close_path(cr);
    cairo_set_source_rgba(cr, 0.85, 0.95, 1.0, 0.18);
    cairo_fill_preserve(cr);
    cairo_set_line_width(cr, 2.5);
    cairo_set_source_rgba(cr, 0.95, 1.0, 1.0, 0.9);
    cairo_stroke(cr);
    cairo_move_to(cr, gx - wt / 2 + 7, y0 + 8);
    cairo_line_to(cr, gx - wb / 2 + 6, y1 - 10);
    cairo_set_line_width(cr, 3);
    cairo_set_source_rgba(cr, 1, 1, 1, 0.55);
    cairo_stroke(cr);
}

static void draw_robot(cairo_t *cr, double t, int idle)
{
    /* body: a graphics card standing on end, with a rounded shell */
    const double bx = 352, by = 262, bw = 132, bh = 150;
    cairo_new_path(cr);
    double r = 22;
    cairo_arc(cr, bx + r, by + r, r, M_PI, 1.5 * M_PI);
    cairo_arc(cr, bx + bw - r, by + r, r, 1.5 * M_PI, 2 * M_PI);
    cairo_arc(cr, bx + bw - r, by + bh - r, r, 0, 0.5 * M_PI);
    cairo_arc(cr, bx + r, by + bh - r, r, 0.5 * M_PI, M_PI);
    cairo_close_path(cr);
    cairo_pattern_t *g = cairo_pattern_create_linear(bx, by, bx + bw, by + bh);
    cairo_pattern_add_color_stop_rgb(g, 0, 0.30, 0.31, 0.38);
    cairo_pattern_add_color_stop_rgb(g, 1, 0.12, 0.12, 0.16);
    cairo_set_source(cr, g);
    cairo_fill_preserve(cr);
    cairo_pattern_destroy(g);
    cairo_set_line_width(cr, 4);
    cairo_set_source_rgb(cr, 0.06, 0.06, 0.09);
    cairo_stroke(cr);
    /* RGB edge: blue to orange, the two GPUs */
    g = cairo_pattern_create_linear(bx, 0, bx + bw, 0);
    cairo_pattern_add_color_stop_rgb(g, 0, 0.25, 0.60, 1.0);
    cairo_pattern_add_color_stop_rgb(g, 1, 1.0, 0.55, 0.15);
    cairo_rectangle(cr, bx + 14, by + 10, bw - 28, 6);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    /* antenna */
    cairo_move_to(cr, bx + 70, by);
    cairo_line_to(cr, bx + 78, by - 26);
    cairo_set_line_width(cr, 3);
    cairo_set_source_rgb(cr, 0.20, 0.20, 0.26);
    cairo_stroke(cr);
    double blink = idle ? 0.3 : 0.6 + 0.4 * sin(t * 6);
    radial_blob(cr, bx + 78, by - 28, 10, (rgb){ 1, 0.3, 0.2 }, blink);
    cairo_arc(cr, bx + 78, by - 28, 4, 0, 2 * M_PI);
    cairo_set_source_rgb(cr, 1, 0.35, 0.25);
    cairo_fill(cr);

    /* fan eyes, glancing toward the glass */
    for (int f = 0; f < 2; f++) {
        double fx = bx + 38 + f * 58, fy = by + 64;
        cairo_arc(cr, fx, fy, 24, 0, 2 * M_PI);
        cairo_set_source_rgb(cr, 0.04, 0.04, 0.06);
        cairo_fill(cr);
        cairo_set_source_rgba(cr, 0.58, 0.60, 0.68, 0.9);
        for (int b = 0; b < 7; b++) {
            double a = fan * 2 * M_PI * (f ? -1 : 1) + b * 2 * M_PI / 7;
            cairo_move_to(cr, fx, fy);
            cairo_arc(cr, fx, fy, 21, a, a + 0.5);
            cairo_close_path(cr);
        }
        cairo_fill(cr);
        rgb eye = idle ? (rgb){ 0.4, 0.4, 0.5 } : (rgb){ 1, 0.28, 0.2 };
        radial_blob(cr, fx - 6, fy + 4, 14, eye, idle ? 0.3 : 0.7);
        cairo_arc(cr, fx - 6, fy + 4, 6, 0, 2 * M_PI);
        set_rgb(cr, eye);
        cairo_fill(cr);
    }
    /* a thin smug slot of a mouth */
    cairo_move_to(cr, bx + 44, by + 106);
    cairo_curve_to(cr, bx + 60, by + 112, bx + 78, by + 112, bx + 92, by + 104);
    cairo_set_line_width(cr, 4);
    cairo_set_source_rgb(cr, 0.06, 0.06, 0.09);
    cairo_stroke(cr);
}

static void update_hud(const stats *s, const shown_t *sh, double t)
{
    const double c = SIZE / 2.0;
    static double tok, watts, ml, next;
    char key[128], txt[64];
    if (t >= next || t < next - 1) {
        next = t + 0.25;
        tok = sh->tok;
        watts = s->power[0] + s->power[1];
        ml = harvested_ml;
    }
    int idle = s->tok_s < 1 && s->running == 0;
    snprintf(key, sizeof(key), "%.0f|%.0f|%.0f|%d", ml, tok, watts, idle);
    if (!strcmp(key, hud_key))
        return;
    strcpy(hud_key, key);
    cairo_t *cr = cairo_create(hud_cache);
    cairo_set_operator(cr, CAIRO_OPERATOR_CLEAR);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    if (ml < 1000)
        snprintf(txt, sizeof(txt), "%.0f mL of tears", ml);
    else
        snprintf(txt, sizeof(txt), "%.2f L of tears", ml / 1000);
    soft_text(cr, c, 50, 30, (rgb){ 0.78, 0.92, 1.0 }, txt);
    if (idle)
        snprintf(txt, sizeof(txt), "idle \xC2\xB7 %.0f W", watts);
    else
        snprintf(txt, sizeof(txt), "%.0f tok/s \xC2\xB7 %.0f W", tok, watts);
    soft_text(cr, c, 452, 22, (rgb){ 1, 1, 1 }, txt);
    cairo_destroy(cr);
}

static void render(cairo_t *cr, const stats *s, const shown_t *sh, double t)
{
    const double c = SIZE / 2.0;
    int idle = s->tok_s < 1 && s->running == 0;

    /* Warm lamplight fading to cold blue as the child gets sadder (cached per mood level) */
    int level = (int)(sadness * 40);
    if (level != bg_level) {
        bg_level = level;
        double sd = level / 40.0;
        cairo_t *bc = cairo_create(bg_cache);
        rgb wall = lerp((rgb){ 0.98, 0.86, 0.66 }, (rgb){ 0.40, 0.48, 0.64 }, sd);
        rgb lamp = lerp((rgb){ 1.0, 0.92, 0.70 }, (rgb){ 0.62, 0.70, 0.86 }, sd);
        set_rgb(bc, wall);
        cairo_paint(bc);
        radial_blob(bc, 90, 70, 260, lamp, 0.9);
        /* storybook wallpaper: soft dots */
        for (int y = 30; y < SIZE; y += 46)
            for (int x = 20 + (y / 46 % 2) * 23; x < SIZE; x += 46) {
                cairo_arc(bc, x, y, 3.5, 0, 2 * M_PI);
                cairo_set_source_rgba(bc, 1, 1, 1, 0.18);
                cairo_fill(bc);
            }
        cairo_pattern_t *g = cairo_pattern_create_radial(c, c, 180, c, c, 250);
        cairo_pattern_add_color_stop_rgba(g, 0, 0.1, 0.05, 0.1, 0);
        cairo_pattern_add_color_stop_rgba(g, 1, 0.1, 0.05, 0.1, 0.55);
        cairo_set_source(bc, g);
        cairo_paint(bc);
        cairo_pattern_destroy(g);
        cairo_destroy(bc);
    }
    cairo_set_source_surface(cr, bg_cache, 0, 0);
    cairo_paint(cr);

    double gx, gy;
    glass_pos(&gx, &gy);

    draw_robot(cr, t, idle);
    draw_child(cr, t);
    for (int i = 0; i < MAX_TEARS; i++)
        if (tears[i].alive)
            draw_tear(cr, &tears[i]);
    draw_glass(cr, gx, gy);
    arm_to(cr, gx + 26, gy + 8);

    /* Slurping: a bendy straw from the lowered glass to the robot's mouth */
    if (slurp_t >= 0 && arm_p < 0.35) {
        cairo_move_to(cr, gx + 4, gy - 10);
        cairo_curve_to(cr, gx + 4, gy - 90, 360, 340, 398, 366);
        cairo_set_line_width(cr, 8);
        cairo_set_source_rgb(cr, 1, 1, 1);
        cairo_stroke_preserve(cr);
        double dashes[2] = { 8, 8 };
        cairo_set_dash(cr, dashes, 2, -t * 60);
        cairo_set_line_width(cr, 6);
        set_rgb(cr, TEAR);
        cairo_stroke(cr);
        cairo_set_dash(cr, NULL, 0, 0);
        soft_text(cr, 410, 228, 26 + 3 * sin(t * 14), (rgb){ 0.72, 0.88, 1.0 }, "SLURP");
    }

    update_hud(s, sh, t);
    cairo_set_source_surface(cr, hud_cache, 0, 0);
    cairo_paint(cr);
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

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);
    srand((unsigned)time(NULL));

    surf = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cr = cairo_create(surf);
    hud_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, SIZE);
    bg_cache = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    tj = tj3Init(TJINIT_COMPRESS);
    tj3Set(tj, TJPARAM_QUALITY, JPEG_QUALITY);
    tj3Set(tj, TJPARAM_SUBSAMP, TJSAMP_420);

    if (bench) {
        struct { double tok; double secs; const char *png; } scenes[] = {
            { 150, 20, "tears_content.png" },
            { 500, 40, "tears_sad.png" },
            { 900, 50, "tears_pout.png" },
            { 1600, 90, "tears_preview.png" },
        };
        for (int k = 0; k < 4; k++) {
            memset(tears, 0, sizeof(tears));
            recent = sadness = glass = harvested_ml = wet = arm_p = 0;
            slurp_t = -1;
            s.tok_port[0] = scenes[k].tok * 0.45; s.tok_port[1] = scenes[k].tok * 0.55;
            s.tok_s = scenes[k].tok; s.running = 4;
            s.power[0] = 25 + scenes[k].tok * 0.28; s.power[1] = 30 + scenes[k].tok * 0.28;
            sh.tok = s.tok_s;
            int n = (int)(scenes[k].secs * FPS_BUSY);
            size_t len = 0;
            double b0 = 0;
            for (int i = 0; i < n; i++) {
                if (i == n - 60)
                    b0 = now_s();
                simulate(&s, 1.0 / FPS_BUSY, i / (double)FPS_BUSY);
                render(cr, &s, &sh, i / (double)FPS_BUSY);
                cairo_surface_flush(surf);
                tj3Compress8(tj, cairo_image_surface_get_data(surf), SIZE, cairo_image_surface_get_stride(surf),
                             SIZE, TJPF_BGRX, &jpeg, &len);
            }
            printf("%s: %.2f ms/frame, sadness %.2f, arm %.2f, glass %.2f\n", scenes[k].png,
                   (now_s() - b0) * 1000 / 60, sadness, arm_p, glass);
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
                vllm_poll(&s, t);
            }
        }

        k = dt * 4 > 1 ? 1 : dt * 4;
        sh.tok += (s.tok_s - sh.tok) * k;

        simulate(&s, dt, t - t0);
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

        int idle_now = s.tok_s < 1 && s.running == 0 && sadness < 0.05 && arm_p < 0.02;
        double spare = 1.0 / (idle_now ? FPS_IDLE : FPS_BUSY) - (now_s() - t);
        if (spare > 0) {
            struct timespec ts = { 0, (long)(spare * 1e9) };
            nanosleep(&ts, NULL);
        }
    }

    tj3Free(jpeg);
    tj3Destroy(tj);
    cairo_destroy(cr);
    cairo_surface_destroy(surf);
    cairo_surface_destroy(hud_cache);
    cairo_surface_destroy(bg_cache);
    if (nvml_ok)
        nvmlShutdown();
    if (fd >= 0)
        close(fd);
    return 0;
}
