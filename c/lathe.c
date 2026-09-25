/*
 * lathe: a record-cutting lathe on the iCUE LINK AIO pump LCD that cuts your LLM's
 * output into a vinyl record, live.
 *
 * The round screen is a spinning lacquer disc. While the vLLM servers generate, the
 * cutter head carves a spiral groove inward, and the groove is the data: a stereo cut
 * where the inner wall (blue) is GPU 0's server and the outer wall (orange) GPU 1's,
 * brightness and wiggle follow each server's tokens/sec, and, like a real variable-pitch
 * lathe, loud passages (high tok/s) get wider groove spacing. Pauses between requests
 * leave the dark gaps you see between tracks on an LP, so the disc is a spiral chart of
 * the last minute or two. When the cutter reaches the label the record flips to the next
 * side. Idle: the needle lifts and the platter spins down.
 * Run with --demo to simulate data, --showcase for a scripted loop, --bench to write
 * preview PNGs.
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
 * Layers: the plinth and platter are drawn once (bg_cache). The vinyl lives in its own
 * surface in the record's frame of reference (disc): new groove is cut into it
 * incrementally, a few short strokes per frame, and it is painted rotated each frame.
 * The anisotropic "bowtie" sheen of a real record doesn't turn with the disc, so it is a
 * static overlay computed once. Label and readout text are cached and redrawn only when
 * the numbers change (4x/s).
 */
#define FPS_BUSY        20
#define FPS_IDLE        15
#define C0              (SIZE / 2.0)
#define R_REC           213.0       /* vinyl edge; the platter rim is outside it */
#define R_OUT           204.0       /* first groove */
#define R_IN            104.0       /* last groove: the side is full */
#define R_LABEL         95.0
#define PITCH_MIN       2.2         /* px per turn when quiet (variable-pitch cutting) */
#define PITCH_MAX       4.8         /* px per turn at TOK_FULL */
#define TOK_FULL        1600.0      /* total tok/s for the widest groove spacing */
#define CH_FULL         800.0       /* per-server tok/s for a full-volume channel */
#define RPM_UP          1200.0      /* switch to 45 rpm above this total tok/s ... */
#define RPM_DOWN        900.0       /* ... and back to 33 1/3 below this */
#define GAP_HOLD        6.0         /* s of silence cut as a track gap before the needle lifts */
#define TRACK_GAP       2.0         /* s of silence that starts a new track */
#define PIVOT_X         392.0       /* tonearm pivot (top right) */
#define PIVOT_Y         64.0
#define ARM_L           200.0
#define MAX_CHIPS       80

typedef struct { double x, y; } pt;

static const rgb CREAM = { 0.98, 0.93, 0.82 };

static struct {
    double rot, omega;          /* platter angle (rad, clockwise) and speed (rad/s) */
    double r_cut;               /* radius the cutter is at */
    double lift;                /* 0 = stylus in the lacquer, 1 = lifted */
    double arm_r;               /* radius the arm is shown at (moves back out on a flip) */
    double silent_t;            /* s since the last activity */
    double flip;                /* <0: not flipping, else 0..1 */
    double ch[2];               /* smoothed per-server tok/s */
    double tot;                 /* smoothed total tok/s */
    double phase;               /* groove modulation phase */
    double prev_th, prev_r;     /* last cut point, record frame */
    int    have_prev, cutting, rpm45, side, track, label_dirty;
    double feed;                /* groove pitch multiplier (showcase speeds it up) */
} dk = { .r_cut = R_OUT, .lift = 1, .arm_r = R_OUT, .silent_t = 99, .flip = -1, .feed = 1 };

typedef struct { double x, y, vx, vy, life; int ch; } chip;
static chip chips[MAX_CHIPS];
static double chip_acc;

static cairo_surface_t *bg_cache, *disc, *sheen_cache, *label_txt, *pill_txt;
static cairo_t *disc_cr;

static double frand(void) { return rand() / (double)RAND_MAX; }

static rgb chan_color(int c) { return c == 0 ? BLUE : ORANGE; }

/* Where the stylus sits when the cutter is at radius r (the arm swings on its pivot) */
static pt arm_tip(double r)
{
    double dx = C0 - PIVOT_X, dy = C0 - PIVOT_Y, d = hypot(dx, dy);
    double ux = dx / d, uy = dy / d;
    double a = (ARM_L * ARM_L - r * r + d * d) / (2 * d);
    double h = sqrt(fmax(0, ARM_L * ARM_L - a * a));
    return (pt){ PIVOT_X + a * ux + h * uy, PIVOT_Y + a * uy - h * ux };
}

/* ---------------------------------------------------------------- static layers */

static void text_path_center(cairo_t *cr, double x, double y, double size, const char *s)
{
    cairo_text_extents_t ext;
    cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, size);
    cairo_text_extents(cr, s, &ext);
    cairo_new_path(cr);
    cairo_move_to(cr, x - ext.width / 2 - ext.x_bearing, y - ext.height / 2 - ext.y_bearing);
    cairo_text_path(cr, s);
}

/* Bold text with a dark outline, centred on (x, y) */
static void text_outline(cairo_t *cr, double x, double y, double size, rgb c, double a, const char *s)
{
    text_path_center(cr, x, y, size, s);
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
    cairo_set_line_width(cr, fmax(3, size * 0.17));
    cairo_set_source_rgba(cr, 0.05, 0.02, 0.02, 0.85 * a);
    cairo_stroke_preserve(cr);
    cairo_set_source_rgba(cr, c.r, c.g, c.b, a);
    cairo_fill(cr);
}

/* Text set along a circle, clockwise from angle a0, glyph by glyph */
static void text_on_circle(cairo_t *cr, double r, double a0, double size, const char *s)
{
    char g[8];
    cairo_text_extents_t ext;
    double a = a0;
    cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, size);
    for (const char *p = s; *p;) {
        int n = 1;                                  /* one UTF-8 sequence per glyph */
        while (n < 7 && ((unsigned char)p[n] & 0xC0) == 0x80)
            n++;
        memcpy(g, p, n);
        g[n] = 0;
        p += n;
        cairo_text_extents(cr, g, &ext);
        double w = ext.x_advance / r;
        cairo_save(cr);
        cairo_translate(cr, C0 + r * cos(a + w / 2), C0 + r * sin(a + w / 2));
        cairo_rotate(cr, a + w / 2 + M_PI / 2);
        cairo_move_to(cr, -ext.x_advance / 2, size * 0.36);
        cairo_show_text(cr, g);
        cairo_restore(cr);
        a += w;
    }
}

/* The record label, drawn into the disc (record frame) */
static void draw_label(cairo_t *cr)
{
    char ring[160];
    cairo_pattern_t *g = cairo_pattern_create_radial(C0 - 30, C0 - 30, 10, C0, C0, R_LABEL);
    cairo_pattern_add_color_stop_rgb(g, 0, 0.66, 0.10, 0.12);
    cairo_pattern_add_color_stop_rgb(g, 1, 0.44, 0.05, 0.08);
    cairo_arc(cr, C0, C0, R_LABEL, 0, 2 * M_PI);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);

    /* printed rings */
    cairo_set_source_rgba(cr, CREAM.r, CREAM.g, CREAM.b, 0.75);
    cairo_set_line_width(cr, 1.6);
    cairo_arc(cr, C0, C0, R_LABEL - 4, 0, 2 * M_PI);
    cairo_stroke(cr);
    cairo_set_line_width(cr, 1.0);
    cairo_arc(cr, C0, C0, 77, 0, 2 * M_PI);
    cairo_stroke(cr);

    snprintf(ring, sizeof(ring), "LLM RECORDS  \xE2\x80\xA2  SIDE %c  \xE2\x80\xA2  TRACK %02d  \xE2\x80\xA2  "
             "STEREO  \xE2\x80\xA2  CUT LIVE  \xE2\x80\xA2  ", 'A' + dk.side % 26, dk.track);
    cairo_set_source_rgba(cr, CREAM.r, CREAM.g, CREAM.b, 0.9);
    text_on_circle(cr, 83.5, -M_PI / 2 - 0.3, 10.5, ring);

    /* spindle */
    cairo_arc(cr, C0, C0, 4.5, 0, 2 * M_PI);
    cairo_set_source_rgb(cr, 0.05, 0.02, 0.03);
    cairo_fill(cr);
}

/* A blank lacquer side: black disc with a bevelled edge and the label */
static void new_side(void)
{
    cairo_t *cr = disc_cr;
    cairo_set_source_rgb(cr, 0.02, 0.02, 0.025);
    cairo_paint(cr);
    cairo_arc(cr, C0, C0, R_REC, 0, 2 * M_PI);
    cairo_set_source_rgb(cr, 0.045, 0.045, 0.055);
    cairo_fill(cr);
    cairo_set_line_width(cr, 2.5);
    cairo_arc(cr, C0, C0, R_REC - 1.5, 0, 2 * M_PI);
    cairo_set_source_rgb(cr, 0.13, 0.13, 0.15);
    cairo_stroke(cr);
    /* run-out area inside the last groove */
    cairo_set_line_width(cr, 1);
    cairo_arc(cr, C0, C0, R_IN - 3, 0, 2 * M_PI);
    cairo_set_source_rgb(cr, 0.09, 0.09, 0.10);
    cairo_stroke(cr);
    draw_label(cr);
    dk.r_cut = R_OUT;
    dk.have_prev = 0;
}

static void build_caches(void)
{
    /* plinth and platter */
    bg_cache = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cairo_t *cr = cairo_create(bg_cache);
    cairo_set_source_rgb(cr, 0.03, 0.03, 0.035);
    cairo_paint(cr);
    cairo_pattern_t *g = cairo_pattern_create_linear(0, 0, SIZE, SIZE);
    cairo_pattern_add_color_stop_rgb(g, 0, 0.36, 0.37, 0.40);
    cairo_pattern_add_color_stop_rgb(g, 0.5, 0.17, 0.17, 0.19);
    cairo_pattern_add_color_stop_rgb(g, 1, 0.30, 0.31, 0.34);
    cairo_arc(cr, C0, C0, 240, 0, 2 * M_PI);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    cairo_set_line_width(cr, 1);
    for (double r = 216; r < 240; r += 2.5) {
        cairo_arc(cr, C0, C0, r, 0, 2 * M_PI);
        cairo_set_source_rgba(cr, 0, 0, 0, 0.12);
        cairo_stroke(cr);
    }
    /* the disc's shadow on the platter */
    g = cairo_pattern_create_radial(C0 + 3, C0 + 4, R_REC - 2, C0 + 3, C0 + 4, R_REC + 9);
    cairo_pattern_add_color_stop_rgba(g, 0, 0, 0, 0, 0.8);
    cairo_pattern_add_color_stop_rgba(g, 1, 0, 0, 0, 0);
    cairo_arc(cr, C0, C0, R_REC + 10, 0, 2 * M_PI);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    /* spindle, seen only while the record is turned over */
    cairo_arc(cr, C0, C0, 6, 0, 2 * M_PI);
    cairo_set_source_rgb(cr, 0.70, 0.71, 0.74);
    cairo_fill(cr);
    cairo_destroy(cr);

    /*
     * Static sheen: light catching the concentric grooves makes two opposite bright
     * wedges that stay put while the record turns, plus fine ring texture.
     */
    sheen_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, SIZE);
    cairo_surface_flush(sheen_cache);
    unsigned char *px = cairo_image_surface_get_data(sheen_cache);
    int stride = cairo_image_surface_get_stride(sheen_cache);
    const double th0 = -0.85;
    for (int y = 0; y < SIZE; y++) {
        uint32_t *row = (uint32_t *)(px + y * stride);
        for (int x = 0; x < SIZE; x++) {
            double dx = x + 0.5 - C0, dy = y + 0.5 - C0, r = hypot(dx, dy), a = 0;
            if (r > R_REC || r < 6) {
                row[x] = 0;
                continue;
            }
            double th = atan2(dy, dx);
            double w = pow(fabs(cos(th - th0)), 18) + 0.35 * pow(fabs(cos(th - th0 - 0.5)), 60);
            if (r > R_LABEL + 1) {
                a = 0.02 + 0.17 * w * (0.85 + 0.15 * sin(r * 0.21));
                if (r > R_REC - 3)
                    a += 0.18 * w + 0.05;
            } else {
                a = 0.10 * w;
            }
            unsigned v = (unsigned)(fmin(1, a) * 255 + 0.5);
            row[x] = (v << 24) | (v << 16) | (v << 8) | v;
        }
    }
    cairo_surface_mark_dirty(sheen_cache);

    disc = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    disc_cr = cairo_create(disc);
    cairo_set_line_cap(disc_cr, CAIRO_LINE_CAP_ROUND);
    cairo_set_line_join(disc_cr, CAIRO_LINE_JOIN_ROUND);
    new_side();

    label_txt = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, 200, 200);
    pill_txt = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, 290, 50);
}

/* ---------------------------------------------------------------- cutting */

static double pitch_now(void)
{
    return dk.feed * (PITCH_MIN + (PITCH_MAX - PITCH_MIN) * pow(clamp01(dk.tot / TOK_FULL), 0.7));
}

/* Cut the groove from the previous point to the stylus's current position */
static void cut_to(double th, double r, double pitch)
{
    cairo_t *cr = disc_cr;
    double th0 = dk.prev_th, r0 = dk.prev_r, dth = remainder(th - th0, 2 * M_PI);
    int n = (int)ceil(fabs(dth) / 0.03) + 1;
    double vol[2], ph0 = dk.phase, arc = fabs(dth) * (r + r0) / 2;

    for (int c = 0; c < 2; c++)
        vol[c] = clamp01(dk.ch[c] / CH_FULL);
    dk.phase += arc / 9.0 * 2 * M_PI;             /* one wiggle per 9 px of groove */

    if (vol[0] < 0.02 && vol[1] < 0.02) {
        /* silence: a plain, dark, unmodulated groove */
        cairo_move_to(cr, C0 + r0 * cos(th0), C0 + r0 * sin(th0));
        for (int j = 1; j <= n; j++) {
            double f = (double)j / n, rr = r0 + (r - r0) * f, tt = th0 + dth * f;
            cairo_line_to(cr, C0 + rr * cos(tt), C0 + rr * sin(tt));
        }
        cairo_set_line_width(cr, 1.0);
        cairo_set_source_rgb(cr, 0.16, 0.16, 0.18);
        cairo_stroke(cr);
        return;
    }

    /* stereo: inner wall = server 0 (blue), outer wall = server 1 (orange) */
    for (int c = 0; c < 2; c++) {
        double v = vol[c], off = (c == 0 ? -1 : 1) * pitch * 0.24;
        double amp = pitch * 0.13 * sqrt(v);
        for (int j = 0; j <= n; j++) {
            double f = (double)j / n, rr = r0 + (r - r0) * f + off, tt = th0 + dth * f;
            rr += amp * sin(ph0 + (dk.phase - ph0) * f + c * 1.9);
            if (j == 0)
                cairo_move_to(cr, C0 + rr * cos(tt), C0 + rr * sin(tt));
            else
                cairo_line_to(cr, C0 + rr * cos(tt), C0 + rr * sin(tt));
        }
        rgb col = v < 0.02 ? (rgb){ 0.16, 0.16, 0.18 } : lerp((rgb){ 0.16, 0.16, 0.18 }, chan_color(c), 0.5 + 0.5 * pow(v, 0.5));
        cairo_set_line_width(cr, v < 0.02 ? 0.9 : 1.0 + pitch * 0.2 * sqrt(v));
        set_rgb(cr, col);
        cairo_stroke(cr);
    }
}

static void spawn_chips(double dt, pt tip)
{
    chip_acc += fmin(dk.tot, 2000) / 30.0 * dt;
    while (chip_acc >= 1) {
        chip_acc -= 1;
        for (int i = 0; i < MAX_CHIPS; i++) {
            chip *p = &chips[i];
            if (p->life > 0)
                continue;
            /* sprayed sideways out of the cutter, dragged along by the turning disc */
            double rx = tip.x - C0, ry = tip.y - C0, l = hypot(rx, ry);
            double side = frand() < 0.5 ? -1 : 1, sp = 60 + frand() * 110, drag = 0.3 + frand() * 0.6;
            rx /= l;
            ry /= l;
            p->x = tip.x;
            p->y = tip.y;
            p->vx = (side * rx - drag * ry) * sp;
            p->vy = (side * ry + drag * rx) * sp;
            p->life = 0.35 + frand() * 0.3;
            p->ch = frand() * (dk.ch[0] + dk.ch[1] + 1e-6) < dk.ch[0] ? 0 : 1;
            break;
        }
    }
}

/* Returns nonzero while anything is moving */
static int simulate(const stats *s, double dt)
{
    int active = s->tok_s >= 1 || s->running > 0, moving = 0;
    double k = fmin(1, dt * 3);

    dk.ch[0] += (s->tok_port[0] - dk.ch[0]) * k;
    dk.ch[1] += (s->tok_port[1] - dk.ch[1]) * k;
    dk.tot = dk.ch[0] + dk.ch[1];

    if (active) {
        if (dk.silent_t >= TRACK_GAP) {
            dk.track++;
            dk.label_dirty = 1;
        }
        dk.silent_t = 0;
    } else {
        dk.silent_t += dt;
    }
    if (s->tok_s > RPM_UP)
        dk.rpm45 = 1;
    else if (s->tok_s < RPM_DOWN)
        dk.rpm45 = 0;

    double want_w = 0, want_lift = 1, want_arm = dk.r_cut;
    double play_w = (dk.rpm45 ? 45.0 : 100.0 / 3) * 2 * M_PI / 60;
    if (dk.flip >= 0) {
        /* changing sides: lift, swing back out, turn the record over */
        if (dk.lift > 0.95) {
            want_arm = R_OUT;
            dk.flip += dt / 1.6;
            if (dk.flip >= 0.5 && dk.flip - dt / 1.6 < 0.5) {
                dk.side++;
                dk.track = active ? 1 : 0;
                new_side();
            }
            if (dk.flip >= 1) {
                dk.flip = -1;
                if (!active)                      /* nothing playing: leave the needle up */
                    dk.silent_t = fmax(dk.silent_t, GAP_HOLD);
            }
        }
        want_w = dk.omega;
    } else if (active || dk.silent_t < GAP_HOLD) {
        want_w = play_w;
        if (dk.omega > 0.85 * play_w && fabs(dk.arm_r - dk.r_cut) < 0.5)
            want_lift = 0;
    } else if (dk.lift < 0.9) {
        want_w = dk.omega;                        /* lift first, then spin down */
    }

    dk.omega += (want_w - dk.omega) * fmin(1, dt * (want_w > dk.omega ? 1.6 : 0.9));
    if (dk.omega < 0.004 && want_w == 0)
        dk.omega = 0;
    dk.lift += (want_lift > dk.lift ? 1 : -1) * fmin(fabs(want_lift - dk.lift), dt * 2.2);
    dk.arm_r += (want_arm > dk.arm_r ? 1 : -1) * fmin(fabs(want_arm - dk.arm_r), dt * 90);

    double dth = dk.omega * dt;
    dk.rot = fmod(dk.rot + dth, 2 * M_PI);

    dk.cutting = dk.flip < 0 && dk.lift < 0.02;
    if (dk.cutting) {
        double pitch = pitch_now();
        double r = dk.r_cut - pitch * dth / (2 * M_PI);
        pt tip = arm_tip(r);
        double th = atan2(tip.y - C0, tip.x - C0) - dk.rot;
        if (dk.have_prev)
            cut_to(th, r, pitch);
        dk.prev_th = th;
        dk.prev_r = r;
        dk.have_prev = 1;
        dk.r_cut = dk.arm_r = r;
        if (dk.tot > 5)
            spawn_chips(dt, tip);
        if (dk.r_cut <= R_IN)
            dk.flip = 0;                          /* side full */
    } else {
        dk.have_prev = 0;
    }

    for (int i = 0; i < MAX_CHIPS; i++) {
        chip *p = &chips[i];
        if (p->life <= 0)
            continue;
        moving = 1;
        p->x += p->vx * dt;
        p->y += p->vy * dt;
        p->vx *= 1 - dt * 3;
        p->vy *= 1 - dt * 3;
        p->life -= dt;
    }

    if (dk.label_dirty) {
        dk.label_dirty = 0;
        draw_label(disc_cr);
    }
    return moving || dk.omega > 0 || dk.flip >= 0 || (dk.lift > 0.001 && dk.lift < 0.999) ||
           fabs(dk.arm_r - dk.r_cut) > 0.01;
}

/* ---------------------------------------------------------------- render */

typedef struct { double zotac, tuf, tok; } shown_t;

static void update_text(const stats *s, const shown_t *sh)
{
    static char last_key[160];
    char key[160], a[32], b[32], c[32], d[32];
    int idle = s->tok_s < 1 && s->running == 0;

    snprintf(a, sizeof(a), "%.0f", shown_gpu_rate(sh->zotac));
    snprintf(b, sizeof(b), "%.0f", shown_gpu_rate(sh->tuf));
    snprintf(c, sizeof(c), "%.0f", shown_rate(sh->tok));
    snprintf(d, sizeof(d), "%s", dk.rpm45 ? "45 RPM" : "33\xE2\x85\x93 RPM");
    snprintf(key, sizeof(key), "%d|%s|%s|%s|%s|%c|%.0f|%d|%d|%d", idle && dk.lift > 0.5, a, b, c, d, 'A' + dk.side % 26,
             s->power[0] + s->power[1], s->temp[0], s->temp[1], dk.flip >= 0);
    if (!strcmp(key, last_key))
        return;
    snprintf(last_key, sizeof(last_key), "%s", key);

    cairo_t *cr = cairo_create(label_txt);
    cairo_set_operator(cr, CAIRO_OPERATOR_CLEAR);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    char side[16];
    snprintf(side, sizeof(side), "SIDE %c", 'A' + dk.side % 26);
    if (dk.flip >= 0) {
        text_outline(cr, 100, 100, 26, CREAM, 1, "flip!");
    } else if (idle && dk.lift > 0.5) {
        text_outline(cr, 100, 96, 38, CREAM, 0.9, "idle");
        text_outline(cr, 100, 134, 17, CREAM, 0.8, side);
    } else {
        text_outline(cr, 66, 60, 25, lerp(BLUE, WHITE, 0.25), 1, a);
        text_outline(cr, 134, 60, 25, lerp(ORANGE, WHITE, 0.2), 1, b);
        text_outline(cr, 100, 104, shown_rate(sh->tok) >= 999.5 ? 44 : 50, WHITE, 1, c);
        text_outline(cr, 100, 137, 18, CREAM, 0.95, rate_unit());
        text_outline(cr, 100, 159, 15, CREAM, 0.8, d);
    }
    cairo_destroy(cr);

    cr = cairo_create(pill_txt);
    cairo_set_operator(cr, CAIRO_OPERATOR_CLEAR);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    cairo_new_path(cr);
    cairo_arc(cr, 25, 25, 22, M_PI / 2, 3 * M_PI / 2);
    cairo_arc(cr, 265, 25, 22, -M_PI / 2, M_PI / 2);
    cairo_close_path(cr);
    cairo_set_source_rgba(cr, 0.02, 0.02, 0.03, 0.72);
    cairo_fill_preserve(cr);
    cairo_set_source_rgba(cr, 1, 1, 1, 0.15);
    cairo_set_line_width(cr, 1.2);
    cairo_stroke(cr);
    snprintf(a, sizeof(a), "%d\xC2\xB0", s->temp[0]);
    snprintf(b, sizeof(b), "%d\xC2\xB0", s->temp[1]);
    snprintf(c, sizeof(c), "%.0f W", s->power[0] + s->power[1]);
    text_outline(cr, 48, 25, 22, BLUE, 1, a);
    text_outline(cr, 145, 25, 24, WHITE, 1, c);
    text_outline(cr, 242, 25, 22, ORANGE, 1, b);
    cairo_destroy(cr);
}

/* Tonearm + cutter head; dx/dy offset and colour alpha let it double as its own shadow */
static void draw_arm(cairo_t *cr, pt tip, double ox, double oy, int shadow)
{
    double ang = atan2(tip.y - PIVOT_Y, tip.x - PIVOT_X);
    pt base = { tip.x - cos(ang) * 46, tip.y - sin(ang) * 46 };

    cairo_save(cr);
    cairo_translate(cr, ox, oy);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);

    /* tube */
    cairo_move_to(cr, PIVOT_X, PIVOT_Y);
    cairo_line_to(cr, base.x, base.y);
    if (shadow) {
        cairo_set_source_rgba(cr, 0, 0, 0, 0.45);
        cairo_set_line_width(cr, 10);
        cairo_stroke(cr);
    } else {
        cairo_set_source_rgb(cr, 0.16, 0.16, 0.18);
        cairo_set_line_width(cr, 10);
        cairo_stroke_preserve(cr);
        cairo_set_source_rgb(cr, 0.72, 0.73, 0.76);
        cairo_set_line_width(cr, 6);
        cairo_stroke_preserve(cr);
        cairo_set_source_rgb(cr, 0.95, 0.96, 0.98);
        cairo_set_line_width(cr, 1.6);
        cairo_stroke(cr);
    }

    /* cutter head, in the arm's frame: x along the arm, stylus at x = 0 */
    cairo_save(cr);
    cairo_translate(cr, tip.x, tip.y);
    cairo_rotate(cr, ang);
    cairo_new_path(cr);
    cairo_move_to(cr, -50, -15);
    cairo_line_to(cr, -6, -15);
    cairo_arc(cr, -6, -7, 8, -M_PI / 2, 0);
    cairo_line_to(cr, 2, 7);
    cairo_arc(cr, -6, 7, 8, 0, M_PI / 2);
    cairo_line_to(cr, -50, 15);
    cairo_close_path(cr);
    if (shadow) {
        cairo_set_source_rgba(cr, 0, 0, 0, 0.45);
        cairo_fill(cr);
    } else {
        cairo_pattern_t *g = cairo_pattern_create_linear(0, -15, 0, 15);
        cairo_pattern_add_color_stop_rgb(g, 0, 0.42, 0.43, 0.47);
        cairo_pattern_add_color_stop_rgb(g, 0.45, 0.20, 0.20, 0.23);
        cairo_pattern_add_color_stop_rgb(g, 1, 0.10, 0.10, 0.12);
        cairo_set_source(cr, g);
        cairo_fill_preserve(cr);
        cairo_pattern_destroy(g);
        cairo_set_source_rgb(cr, 0.05, 0.05, 0.06);
        cairo_set_line_width(cr, 1.5);
        cairo_stroke(cr);
        /* cooling fins */
        cairo_set_source_rgba(cr, 0, 0, 0, 0.5);
        cairo_set_line_width(cr, 1.5);
        for (int i = 0; i < 4; i++) {
            cairo_move_to(cr, -44 + i * 6, -11);
            cairo_line_to(cr, -44 + i * 6, 11);
        }
        cairo_stroke(cr);
        /* REC lamp */
        cairo_arc(cr, -14, 0, 4.5, 0, 2 * M_PI);
        if (dk.cutting && dk.tot > 5)
            cairo_set_source_rgb(cr, 1.0, 0.18, 0.15);
        else
            cairo_set_source_rgb(cr, 0.30, 0.06, 0.06);
        cairo_fill(cr);
    }
    cairo_restore(cr);

    /* pivot */
    if (!shadow) {
        cairo_arc(cr, PIVOT_X, PIVOT_Y, 26, 0, 2 * M_PI);
        cairo_set_source_rgb(cr, 0.12, 0.12, 0.14);
        cairo_fill(cr);
        cairo_arc(cr, PIVOT_X, PIVOT_Y, 13, 0, 2 * M_PI);
        cairo_set_source_rgb(cr, 0.62, 0.63, 0.66);
        cairo_fill(cr);
    }
    cairo_restore(cr);
}

static void render(cairo_t *cr, const stats *s, const shown_t *sh)
{
    double xs = dk.flip >= 0 ? fabs(cos(M_PI * dk.flip)) : 1;

    cairo_set_source_surface(cr, bg_cache, 0, 0);
    cairo_paint(cr);

    /* strobe dots on the platter rim: at exactly 33 1/3 rpm and 20 fps they stand still */
    cairo_set_source_rgba(cr, 0.85, 0.86, 0.9, 0.55);
    for (int i = 0; i < 36; i++) {
        double a = dk.rot + i * (2 * M_PI / 36);
        cairo_new_sub_path(cr);
        cairo_arc(cr, C0 + 228 * cos(a), C0 + 228 * sin(a), 2.6, 0, 2 * M_PI);
    }
    cairo_fill(cr);

    /* the record, turned (and during a flip, squashed) */
    cairo_save(cr);
    cairo_translate(cr, C0, C0);
    cairo_scale(cr, fmax(xs, 0.02), 1);
    cairo_arc(cr, 0, 0, R_REC, 0, 2 * M_PI);
    cairo_clip(cr);
    cairo_rotate(cr, dk.rot);
    cairo_translate(cr, -C0, -C0);
    cairo_set_source_surface(cr, disc, 0, 0);
    cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_BILINEAR);
    cairo_paint(cr);
    cairo_restore(cr);

    if (dk.flip < 0) {
        cairo_set_source_surface(cr, sheen_cache, 0, 0);
        cairo_paint(cr);
    }

    pt tip = arm_tip(dk.arm_r);
    double th = atan2(tip.y - C0, tip.x - C0);

    /* hot, freshly cut groove glowing behind the stylus */
    if (dk.cutting && dk.tot > 5) {
        double heat = clamp01(dk.tot / TOK_FULL), f0 = dk.ch[0] / (dk.tot + 1e-6);
        rgb col = lerp(lerp(ORANGE, BLUE, f0), WHITE, 0.4);
        cairo_set_line_cap(cr, CAIRO_LINE_CAP_BUTT);
        for (int i = 0; i < 10; i++) {
            cairo_new_path(cr);
            cairo_arc(cr, C0, C0, dk.arm_r, th + i * 0.09, th + (i + 1) * 0.09 + 0.01);
            cairo_set_source_rgba(cr, col.r, col.g, col.b, (0.45 + 0.5 * heat) * (1 - i / 10.0));
            cairo_set_line_width(cr, 3 + pitch_now() * 1.1);
            cairo_stroke(cr);
        }
        cairo_pattern_t *g = cairo_pattern_create_radial(tip.x, tip.y, 0, tip.x, tip.y, 16 + 10 * heat);
        cairo_pattern_add_color_stop_rgba(g, 0, 1, 0.95, 0.85, 0.9);
        cairo_pattern_add_color_stop_rgba(g, 0.35, col.r, col.g, col.b, 0.45);
        cairo_pattern_add_color_stop_rgba(g, 1, col.r, col.g, col.b, 0);
        cairo_set_source(cr, g);
        cairo_arc(cr, tip.x, tip.y, 26, 0, 2 * M_PI);
        cairo_fill(cr);
        cairo_pattern_destroy(g);
    }

    /* lacquer chips thrown off the cutter */
    for (int c = 0; c < 2; c++) {
        rgb col = lerp(chan_color(c), WHITE, 0.3);
        int any = 0;
        for (int i = 0; i < MAX_CHIPS; i++) {
            const chip *p = &chips[i];
            if (p->life <= 0 || p->ch != c)
                continue;
            cairo_rectangle(cr, p->x - 1.3, p->y - 1.3, 2.6, 2.6);
            any = 1;
        }
        if (any) {
            cairo_set_source_rgba(cr, col.r, col.g, col.b, 0.85);
            cairo_fill(cr);
        }
    }

    /* arm: the shadow moves away as the head lifts */
    double lift = dk.lift * dk.lift * (3 - 2 * dk.lift);
    draw_arm(cr, tip, 5 + 11 * lift, 7 + 15 * lift, 1);
    draw_arm(cr, tip, 0, 0, 0);

    update_text(s, sh);
    cairo_set_source_surface(cr, label_txt, C0 - 100, C0 - 100);
    cairo_paint(cr);
    cairo_set_source_surface(cr, pill_txt, C0 - 145, 372);
    cairo_paint(cr);
}

/* ---------------------------------------------------------------- showcase */

/*
 * --showcase: a scripted 44 s loop for filming or GIFs: a quiet track on GPU 0 only,
 * a gap, a loud track on both, a gap, GPU 1 alone, then more load until the side is
 * full and the record flips. The groove pitch is 1.3x so a side fills in time.
 */
#define SHOWCASE_LEN    44.0

static void showcase_poll(stats *s, double t)
{
    double lt = fmod(t, SHOWCASE_LEN), a = 0, b = 0;
    if (lt < 2.5) {
    } else if (lt < 10) {
        a = 230 + 30 * sin(lt * 2.1);
    } else if (lt < 13) {
    } else if (lt < 24) {
        a = 760 + 60 * sin(lt * 1.3);
        b = 820 + 70 * sin(lt * 0.9 + 1);
    } else if (lt < 26.5) {
    } else if (lt < 33) {
        b = 420 + 50 * sin(lt * 1.7);
    } else if (lt < 40) {
        a = 600 + 80 * sin(lt * 1.1);
        b = 650;
    }
    s->tok_port[0] = a;
    s->tok_port[1] = b;
    s->tok_s = a + b;
    s->running = s->tok_s > 0 ? 1 + (int)(s->tok_s / 250) : 0;
    s->power[0] = 30 + fmin(545, a * 0.62);
    s->power[1] = 35 + fmin(540, b * 0.6);
    s->temp[0] = (int)(38 + a / 40);
    s->temp[1] = (int)(44 + b / 34);
}

/* ---------------------------------------------------------------- main */

static void ease_shown(shown_t *sh, const stats *s, double dt)
{
    double k = dt * 4 > 1 ? 1 : dt * 4;
    sh->tok += (s->tok_s - sh->tok) * k;
    sh->zotac += (s->tok_port[0] - sh->zotac) * k;
    sh->tuf += (s->tok_port[1] - sh->tuf) * k;
}

int main(int argc, char **argv)
{
    cairo_surface_t *surf;
    cairo_t *cr;
    tjhandle tj;
    unsigned char *jpeg = NULL;
    stats s = { 0 };
    shown_t sh = { 0 }, sh_view = { 0 };
    double last, next_poll = 0, next_text = 0, t0;
    int fd = -1, bench = 0, showcase = 0, moving = 1;

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
    srand((unsigned)time(NULL));
    build_caches();

    surf = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cr = cairo_create(surf);
    tj = tj3Init(TJINIT_COMPRESS);
    tj3Set(tj, TJPARAM_QUALITY, JPEG_QUALITY);
    tj3Set(tj, TJPARAM_SUBSAMP, TJSAMP_420);

    if (bench) {
        /* each scene replays some history (seconds, tok/s per server) to cut grooves, then times 60 frames */
        struct seg { double secs, a, b; };
        static const struct seg heavy[] = { { 6, 220, 0 }, { 3, 0, 0 }, { 6, 420, 300 }, { 3, 0, 0 }, { 13, 780, 840 }, { 0, 0, 0 } };
        static const struct seg light[] = { { 7, 700, 760 }, { 3, 0, 0 }, { 9, 0, 330 }, { 3, 0, 0 }, { 22, 190, 120 }, { 0, 0, 0 } };
        static const struct seg idle[] = { { 9, 720, 800 }, { 3, 0, 0 }, { 10, 260, 0 }, { 3, 0, 0 }, { 9, 450, 520 }, { 25, 0, 0 }, { 0, 0, 0 } };
        struct { const struct seg *segs; double w0, w1; int t0, t1; const char *png; } scenes[] = {
            { heavy, 470, 460, 58, 71, "lathe_preview.png" },
            { light, 190, 120, 47, 55, "lathe_light.png" },
            { idle, 28, 32, 36, 41, "lathe_idle.png" },
        };
        for (int k = 0; k < 3; k++) {
            double dt = 1.0 / FPS_BUSY;
            int nfr = 0, total = 0;
            size_t len = 0;
            double b0 = 0;
            memset(chips, 0, sizeof(chips));
            dk.side = k;
            dk.track = 0;
            dk.silent_t = 99;
            new_side();
            for (const struct seg *g = scenes[k].segs; g->secs > 0; g++)
                total += (int)(g->secs * FPS_BUSY);
            for (const struct seg *g = scenes[k].segs; g->secs > 0; g++) {
                for (int i = 0; i < (int)(g->secs * FPS_BUSY); i++, nfr++) {
                    s.tok_port[0] = g->a * (1 + 0.08 * sin(nfr * 0.13));
                    s.tok_port[1] = g->b * (1 + 0.08 * sin(nfr * 0.07 + 1));
                    s.tok_s = s.tok_port[0] + s.tok_port[1];
                    s.running = s.tok_s > 0 ? 4 : 0;
                    s.power[0] = scenes[k].w0; s.power[1] = scenes[k].w1;
                    s.temp[0] = scenes[k].t0; s.temp[1] = scenes[k].t1;
                    if (nfr == total - 60)
                        b0 = now_s();
                    simulate(&s, dt);
                    ease_shown(&sh, &s, dt);
                    if (nfr >= total - 60 || nfr % 20 == 0) {
                        render(cr, &s, &sh);
                        cairo_surface_flush(surf);
                        tj3Compress8(tj, cairo_image_surface_get_data(surf), SIZE, cairo_image_surface_get_stride(surf),
                                     SIZE, TJPF_BGRX, &jpeg, &len);
                    }
                }
            }
            printf("%s: %.2f ms/frame, jpeg %zu bytes\n", scenes[k].png, (now_s() - b0) * 1000 / 60, len);
            cairo_surface_write_to_png(surf, scenes[k].png);
        }
        return 0;
    }

    if (showcase)
        dk.feed = 1.3;
    if (!demo)
        gpus_init();

    while ((fd = lcd_open()) < 0 && !stop)
        sleep(2);
    if (fd >= 0)
        lcd_brightness(fd, 100);

    t0 = last = now_s();
    while (!stop) {
        double t = now_s(), dt = t - last;
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

        ease_shown(&sh, &s, dt);
        if (t >= next_text) {                     /* numbers update 4x per second */
            next_text = t + 0.25;
            sh_view = sh;
        }

        moving = simulate(&s, dt);
        render(cr, &s, &sh_view);
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

        /* Slow down when the platter has stopped */
        int idle = s.tok_s < 1 && s.running == 0 && !moving;
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
    cairo_destroy(disc_cr);
    cairo_surface_destroy(disc);
    cairo_surface_destroy(bg_cache);
    cairo_surface_destroy(sheen_cache);
    cairo_surface_destroy(label_txt);
    cairo_surface_destroy(pill_txt);
    if (nvml_ok)
        nvmlShutdown();
    if (fd >= 0)
        close(fd);
    return 0;
}
