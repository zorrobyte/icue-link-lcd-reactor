/*
 * butwait: a reasoning model is asked "how many r's in strawberry?" on the iCUE LINK
 * AIO pump LCD, and your GPUs overthink it at your real tokens/sec.
 *
 * A strawberry sits under the question and thinks out loud: every ~120 generated
 * tokens becomes one word of its <think> stream (blue from GPU 0's vLLM server, orange
 * from GPU 1's). The faster your servers go, the harder it spirals, from a calm
 * "Hmm. Let me count." through "Wait," and "BUT WAIT" to a sweating, steaming
 * meltdown, and the answer gets worse: 3 when slow, 2 (100% sure) when fast. Every
 * so often it stamps a FINAL ANSWER... and then says "Wait,". The counter is the real
 * number of tokens your servers have generated, all of them spent on this question.
 * Run with --demo to simulate data, --showcase for a scripted loop of every stage,
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
    double tok_total;           /* generation tokens since the servers started */
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
    static double total_seen[N_PORTS];     /* last value per server, kept while one is down */
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
        total_seen[i] = tokens;
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
    s->tok_total = 0;
    for (int i = 0; i < N_PORTS; i++)
        s->tok_total += total_seen[i];
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
 * The background (with the question), every word of the think stream, the stamp and
 * the red heat glow are rendered once into cached surfaces. Each frame blits them,
 * draws the strawberry sprite with a small vector face on top, and pastes the HUD,
 * which is only redrawn when its text changes (numbers update 4 times a second).
 */
#define FPS_BUSY        20
#define FPS_IDLE        8
#define TOKENS_PER_WORD 120.0       /* one word of the think stream per this many tokens */
#define MAX_WORD_RATE   9.0         /* words/sec cap (both servers) so they stay readable */
#define WAIT_TOK        400.0       /* "Wait," takes over */
#define PANIC_TOK       900.0       /* "BUT WAIT", sweating */
#define MELT_TOK        1500.0      /* meltdown: spiral eyes, steam, red glow */
#define STAMP_EVERY     13.0        /* seconds of thinking between FINAL ANSWERs */

#define CX              240.0       /* strawberry centre */
#define CY              206.0
#define HUD_Y           296         /* top of the cached HUD strip */
#define HUD_H           172
#define MAX_WORDS       72
#define MAX_DROPS       32
#define MAX_PUFFS       24

enum { ST_IDLE, ST_THINK, ST_WAIT, ST_PANIC, ST_MELT };
enum { M_SLEEP, M_THINK, M_WAIT, M_PANIC, M_MELT, M_SMUG, M_SHOCK };

typedef struct { cairo_surface_t *s; int w, h; } label;

#define N_VOCAB 8
static const char *vocab[5][N_VOCAB] = {
    { 0 },
    { "Hmm.", "Let me count.", "s-t-r-a-w", "b-e-r-r-y", "r... r...", "So, 3?", "Okay so", "one r, two r" },
    { "Wait,", "Wait,", "But wait,", "Actually,", "Let me recount.", "Hmm, wait.", "Wait,", "Double-check:" },
    { "WAIT", "WAIT,", "BUT WAIT", "HOLD ON", "RECOUNT", "Actually no,", "r? R? rr?", "Wait wait," },
    { "WAIT WAIT", "AAAAA", "strawbery??", "r = 7?", "WAIT", "is r a vowel", "RECOUNT!!", "BUT WAIT" },
};
static const double vocab_size[5] = { 0, 20, 22, 24, 26 };

static const rgb WORD_COL[2] = { { 0.62, 0.86, 1.00 }, { 1.00, 0.76, 0.52 } };
static const rgb YELLOW = { 1.00, 0.86, 0.30 };
static const rgb INK    = { 0.06, 0.02, 0.06 };

static label word_img[5][N_VOCAB][2];
static label think_open, think_close, big_wait, zz[3], stamp_img[2];
static cairo_surface_t *bg_cache, *glow_cache, *berry, *berry_hot, *hud_cache;

typedef struct {
    double x, y, vx, vy, age, life;
    const label *img;
    int alive, top;
} word;

typedef struct { double x, y, vx, vy, age; int alive; } drop;
typedef struct { double x, y, vx, vy, age, life; int alive; } puff;

static word  words[MAX_WORDS];
static drop  drops[MAX_DROPS];
static puff  puffs[MAX_PUFFS];

/* scene state */
static int    stage;
static double stage_since = -10, word_acc[2], burst_acc, drop_acc, puff_acc, z_acc;
static double stamp_start = -100, stamp_next = 8, stamp_answer = 2;
static double wob_amp, heat, blink_next = 2, blink_start = -1, think_start = -100;
static int    stamp_waited, next_vocab;
static int    showcase;

static double frand(void) { return rand() / (double)RAND_MAX; }

/* ---------------------------------------------------------------- caches */

/* Bold text with a dark outline, rendered once into its own surface */
static label make_label(const char *txt, const char *font, double size, rgb fill, double outline)
{
    cairo_surface_t *tmp = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, 1, 1);
    cairo_t *c = cairo_create(tmp);
    cairo_text_extents_t e;
    cairo_select_font_face(c, font, CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(c, size);
    cairo_text_extents(c, txt, &e);
    cairo_destroy(c);
    cairo_surface_destroy(tmp);

    int pad = (int)ceil(outline) + 2;
    label l = { NULL, (int)ceil(e.width) + 2 * pad, (int)ceil(e.height) + 2 * pad };
    l.s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, l.w, l.h);
    c = cairo_create(l.s);
    cairo_select_font_face(c, font, CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(c, size);
    cairo_move_to(c, pad - e.x_bearing, pad - e.y_bearing);
    cairo_text_path(c, txt);
    cairo_set_line_join(c, CAIRO_LINE_JOIN_ROUND);
    cairo_set_line_width(c, outline * 2);
    set_rgb(c, INK);
    cairo_stroke_preserve(c);
    set_rgb(c, fill);
    cairo_fill(c);
    cairo_destroy(c);
    return l;
}

static void paint_label(cairo_t *cr, const label *l, double x, double y, double alpha)
{
    cairo_set_source_surface(cr, l->s, round(x - l->w / 2.0), round(y - l->h / 2.0));
    cairo_paint_with_alpha(cr, alpha);
}

static cairo_surface_t *load_asset(const char *name)
{
    char exe[512], path[1024];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n > 0) {
        exe[n] = 0;
        char *slash = strrchr(exe, '/');
        if (slash)
            *slash = 0;
        snprintf(path, sizeof(path), "%s/assets/butwait/%s", exe, name);
        cairo_surface_t *s = cairo_image_surface_create_from_png(path);
        if (cairo_surface_status(s) == CAIRO_STATUS_SUCCESS)
            return s;
        cairo_surface_destroy(s);
    }
    snprintf(path, sizeof(path), "assets/butwait/%s", name);
    cairo_surface_t *s = cairo_image_surface_create_from_png(path);
    if (cairo_surface_status(s) != CAIRO_STATUS_SUCCESS) {
        fprintf(stderr, "butwait: can't load %s (looked next to the binary and in ./assets/butwait)\n", name);
        exit(1);
    }
    return s;
}

/* Text centred on x at baseline-independent centre y, with outline */
static double text_w(cairo_t *cr, const char *s, double size)
{
    cairo_text_extents_t e;
    cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, size);
    cairo_text_extents(cr, s, &e);
    return e.x_advance;
}

/* Draw text starting at x with its vertical centre on y (cap height based) */
static void text_at(cairo_t *cr, double x, double y, double size, rgb c, const char *s, double outline)
{
    cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, size);
    cairo_move_to(cr, x, y + size * 0.36);
    cairo_text_path(cr, s);
    if (outline > 0) {
        cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
        cairo_set_line_width(cr, outline * 2);
        set_rgb(cr, INK);
        cairo_stroke_preserve(cr);
    }
    set_rgb(cr, c);
    cairo_fill(cr);
}

static void text_mid(cairo_t *cr, double x, double y, double size, rgb c, const char *s, double outline)
{
    text_at(cr, x - text_w(cr, s, size) / 2, y, size, c, s, outline);
}

/* The rubber stamp: FINAL ANSWER: n */
static label make_stamp(const char *digit)
{
    label l = { NULL, 296, 136 };
    const rgb red = { 0.85, 0.08, 0.12 };
    l.s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, l.w, l.h);
    cairo_t *c = cairo_create(l.s);
    double x = 6, y = 6, w = l.w - 12, h = l.h - 12, r = 14;
    cairo_new_sub_path(c);
    cairo_arc(c, x + w - r, y + r, r, -M_PI / 2, 0);
    cairo_arc(c, x + w - r, y + h - r, r, 0, M_PI / 2);
    cairo_arc(c, x + r, y + h - r, r, M_PI / 2, M_PI);
    cairo_arc(c, x + r, y + r, r, M_PI, 1.5 * M_PI);
    cairo_close_path(c);
    cairo_set_source_rgba(c, 1.0, 0.96, 0.88, 0.96);
    cairo_fill_preserve(c);
    set_rgb(c, red);
    cairo_set_line_width(c, 7);
    cairo_stroke(c);
    cairo_rectangle(c, x + 10, y + 10, w - 20, h - 20);
    cairo_set_line_width(c, 2);
    cairo_stroke(c);
    text_mid(c, l.w / 2.0, 38, 28, red, "FINAL ANSWER:", 0);
    text_mid(c, l.w / 2.0, 91, 62, red, digit, 0);
    cairo_destroy(c);
    return l;
}

static void build_caches(void)
{
    const double c = SIZE / 2.0;
    const char *mono = "DejaVu Sans Mono";

    /* think-stream words, one surface per word and server colour */
    for (int st = 1; st < 5; st++)
        for (int i = 0; i < N_VOCAB; i++)
            for (int k = 0; k < 2; k++)
                word_img[st][i][k] = make_label(vocab[st][i], mono, vocab_size[st], WORD_COL[k], 2.5);
    think_open  = make_label("<think>", mono, 22, (rgb){ 0.85, 0.80, 1.0 }, 2.5);
    think_close = make_label("</think>", mono, 22, (rgb){ 0.85, 0.80, 1.0 }, 2.5);
    big_wait    = make_label("Wait,", "DejaVu Sans", 56, YELLOW, 4.5);
    zz[0] = make_label("z", "DejaVu Sans", 24, (rgb){ 0.86, 0.88, 1.0 }, 3);
    zz[1] = make_label("Z", "DejaVu Sans", 30, (rgb){ 0.86, 0.88, 1.0 }, 3);
    zz[2] = make_label("Z", "DejaVu Sans", 36, (rgb){ 0.86, 0.88, 1.0 }, 3);
    stamp_img[0] = make_stamp("3");
    stamp_img[1] = make_stamp("2");

    /* background: dark plum with a soft halo behind the strawberry, and the question */
    bg_cache = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cairo_t *cr = cairo_create(bg_cache);
    cairo_pattern_t *g = cairo_pattern_create_linear(0, 0, 0, SIZE);
    cairo_pattern_add_color_stop_rgb(g, 0.0, 0.10, 0.06, 0.20);
    cairo_pattern_add_color_stop_rgb(g, 0.6, 0.06, 0.04, 0.12);
    cairo_pattern_add_color_stop_rgb(g, 1.0, 0.03, 0.02, 0.06);
    cairo_set_source(cr, g);
    cairo_paint(cr);
    cairo_pattern_destroy(g);
    /* faint notebook grid: it's doing maths, after all */
    cairo_set_source_rgba(cr, 0.55, 0.50, 0.90, 0.07);
    cairo_set_line_width(cr, 1);
    for (int i = 12; i < SIZE; i += 24) {
        cairo_move_to(cr, i + 0.5, 0);
        cairo_line_to(cr, i + 0.5, SIZE);
        cairo_move_to(cr, 0, i + 0.5);
        cairo_line_to(cr, SIZE, i + 0.5);
    }
    cairo_stroke(cr);
    g = cairo_pattern_create_radial(CX, CY + 10, 20, CX, CY + 10, 150);
    cairo_pattern_add_color_stop_rgba(g, 0, 0.95, 0.45, 0.55, 0.30);
    cairo_pattern_add_color_stop_rgba(g, 1, 0.95, 0.45, 0.55, 0);
    cairo_set_source(cr, g);
    cairo_arc(cr, CX, CY + 10, 150, 0, 2 * M_PI);
    cairo_fill(cr);
    cairo_pattern_destroy(g);

    /* the question: "how many r's in" / STRAWBERRY? with the r's picked out */
    text_mid(cr, c, 40, 21, (rgb){ 0.80, 0.78, 0.92 }, "how many r's in", 2.5);
    {
        const char *word = "STRAWBERRY?";
        char ch[2] = { 0, 0 };
        double size = 36, x = c - text_w(cr, word, size) / 2;
        for (const char *p = word; *p; p++) {
            ch[0] = *p;
            int is_r = *p == 'R';
            text_at(cr, x, 76, size, is_r ? (rgb){ 1.0, 0.30, 0.38 } : WHITE, ch, 3);
            x += text_w(cr, ch, size);
        }
    }
    /* round vignette */
    g = cairo_pattern_create_radial(c, c, 200, c, c, 242);
    cairo_pattern_add_color_stop_rgba(g, 0, 0, 0, 0, 0);
    cairo_pattern_add_color_stop_rgba(g, 1, 0, 0, 0, 0.75);
    cairo_set_source(cr, g);
    cairo_paint(cr);
    cairo_pattern_destroy(g);
    cairo_destroy(cr);

    /* red overheat glow at the rim, painted with varying alpha when it's melting down */
    glow_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, SIZE);
    cr = cairo_create(glow_cache);
    g = cairo_pattern_create_radial(c, c, 120, c, c, 240);
    cairo_pattern_add_color_stop_rgba(g, 0, 1.0, 0.15, 0.05, 0);
    cairo_pattern_add_color_stop_rgba(g, 1, 1.0, 0.15, 0.05, 0.65);
    cairo_set_source(cr, g);
    cairo_paint(cr);
    cairo_pattern_destroy(g);
    cairo_destroy(cr);

    /* the strawberry and an overheated copy */
    berry = load_asset("strawberry.png");
    int bw = cairo_image_surface_get_width(berry), bh = cairo_image_surface_get_height(berry);
    berry_hot = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, bw, bh);
    cr = cairo_create(berry_hot);
    cairo_set_source_surface(cr, berry, 0, 0);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_ATOP);
    cairo_set_source_rgba(cr, 1.0, 0.25, 0.05, 0.45);
    cairo_paint(cr);
    cairo_destroy(cr);

    hud_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, HUD_H);
}

/* ---------------------------------------------------------------- simulation */

static void spawn_word(const label *img, double x, double y, double vx, double vy, double life, int top)
{
    for (int i = 0; i < MAX_WORDS; i++)
        if (!words[i].alive) {
            words[i] = (word){ x, y, vx, vy, 0, life, img, 1, top };
            return;
        }
}

/*
 * One word of the think stream. Words slide out from behind the strawberry along
 * horizontal lanes, GPU 0's server to the left and GPU 1's to the right; a lane takes
 * a new word only once the last one has moved clear, so they never pile up.
 */
#define N_LANES 6
static const double lane_y[N_LANES] = { -80, -50, -20, 10, 40, 70 };
static double lane_free[2][N_LANES];

static int spawn_think_word(int server, double t)
{
    int side = server == 0 ? -1 : 1, free_n = 0, pick[N_LANES];
    for (int i = 0; i < N_LANES; i++)
        if (t >= lane_free[server][i])
            pick[free_n++] = i;
    if (!free_n)
        return 0;
    int lane = pick[rand() % free_n];
    const label *img = &word_img[stage][next_vocab % N_VOCAB][server];
    next_vocab = (next_vocab * 5 + 3 + rand() % 3) % 9973;       /* shuffle through the list */
    double v = (60 + frand() * 18) * (1 + 0.25 * (stage - 1));
    lane_free[server][lane] = t + (img->w * (stage == ST_MELT ? 0.8 : 1) + 16) / v;
    spawn_word(img, CX + side * img->w * 0.5, CY + lane_y[lane] + (frand() - 0.5) * 6,
               side * v, -4 - frand() * 6, 4.0, 0);
    return 1;
}

/* Panic and meltdown: extra words blurted out anywhere around the strawberry */
static void spawn_burst_word(int server)
{
    const label *img = &word_img[stage][next_vocab % N_VOCAB][server];
    next_vocab = (next_vocab * 5 + 3 + rand() % 3) % 9973;
    double side = server == 0 ? -1 : 1;
    double x = CX + side * (95 + frand() * 60), y = CY - 85 + frand() * 160;
    spawn_word(img, x, y, side * 20, -12, 1.1, 0);
}

static int stage_for(const stats *s, double tok)
{
    static const double up[5] = { 0, 1, WAIT_TOK, PANIC_TOK, MELT_TOK };
    if (s->tok_s < 1 && s->running == 0)
        return ST_IDLE;
    int st = stage < ST_THINK ? ST_THINK : stage;
    while (st < ST_MELT && tok > up[st + 1] * 1.04)            /* hysteresis both ways */
        st++;
    while (st > ST_THINK && tok < up[st] * 0.90)
        st--;
    return st;
}

static double stamp_age(double t) { return t - stamp_start; }
#define STAMP_IN    0.35            /* </think>, then the stamp slams down */
#define STAMP_HOLD  2.4             /* ...and then "Wait," */
#define STAMP_END   3.1

static int simulate(const stats *s, double tok, double dt, double t)
{
    int want = stage_for(s, tok);
    if (want != stage && t - stage_since > 1.0) {
        if (stage == ST_IDLE) {                                /* woke up: a new question */
            spawn_word(&think_open, CX + 64, CY - 50, 30, -26, 1.8, 0);
            think_start = t;
            if (!showcase)
                stamp_next = t + 7.0;
        }
        stage = want;
        stage_since = t;
    }

    /* the FINAL ANSWER beat */
    double sa = stamp_age(t);
    if (stage >= ST_THINK && t >= stamp_next && sa > STAMP_END) {
        stamp_start = t;
        stamp_answer = stage == ST_THINK ? 3 : 2;
        stamp_waited = 0;
        spawn_word(&think_close, CX + 70, CY - 46, 26, -24, 1.4, 1);
        stamp_next = t + STAMP_EVERY;
        sa = 0;
    }
    if (sa >= STAMP_HOLD && !stamp_waited) {
        stamp_waited = 1;
        spawn_word(&big_wait, CX, CY - 66, 0, -10, 1.7, 1);
    }
    int stamping = sa >= 0 && sa < STAMP_HOLD;

    /* think stream */
    if (stage >= ST_THINK && !stamping) {
        double r[2] = { fmin(s->tok_port[0] / TOKENS_PER_WORD, MAX_WORD_RATE / 2),
                        fmin(s->tok_port[1] / TOKENS_PER_WORD, MAX_WORD_RATE / 2) };
        if (s->tok_s < 1)
            r[0] = 0.5;                                        /* prefill: still mumbling */
        for (int i = 0; i < 2; i++) {
            word_acc[i] = fmin(word_acc[i] + r[i] * dt, 1.5);
            while (word_acc[i] >= 1 && spawn_think_word(i, t))
                word_acc[i] -= 1;
        }
        burst_acc += (stage == ST_MELT ? 5.0 : stage == ST_PANIC ? 1.5 : 0) * dt;
        while (burst_acc >= 1) {
            burst_acc -= 1;
            spawn_burst_word(rand() & 1);
        }
    } else {
        word_acc[0] = word_acc[1] = 0;
    }

    /* idle: Z's */
    if (stage == ST_IDLE) {
        z_acc += dt * 0.8;
        if (z_acc >= 1) {
            z_acc -= 1;
            spawn_word(&zz[rand() % 3], CX + 70 + frand() * 10, CY - 30, 22 + frand() * 10, -20, 3.2, 0);
        }
    }

    /* sweat and steam */
    static const double sweat_rate[5] = { 0, 0, 0.8, 2.6, 4.5 };
    drop_acc += sweat_rate[stage] * dt;
    while (drop_acc >= 1) {
        drop_acc -= 1;
        for (int i = 0; i < MAX_DROPS; i++)
            if (!drops[i].alive) {
                int side = rand() & 1 ? 1 : -1;
                drops[i] = (drop){ CX + side * (40 + frand() * 22), CY - 20 + frand() * 30,
                                   side * (30 + frand() * 40), -(40 + frand() * 50), 0, 1 };
                break;
            }
    }
    puff_acc += (stage == ST_MELT ? 5.0 : 0) * dt;
    while (puff_acc >= 1) {
        puff_acc -= 1;
        for (int i = 0; i < MAX_PUFFS; i++)
            if (!puffs[i].alive) {
                puffs[i] = (puff){ CX + (frand() - 0.5) * 60, CY - 80, (frand() - 0.5) * 30, -(50 + frand() * 30),
                                   0, 1.3 + frand() * 0.5, 1 };
                break;
            }
    }

    int active = 0;
    for (int i = 0; i < MAX_WORDS; i++) {
        word *w = &words[i];
        if (!w->alive)
            continue;
        active++;
        w->age += dt;
        w->x += w->vx * dt;
        w->y += w->vy * dt;
        if (w->age >= w->life || fabs(w->x - CX) - w->img->w / 2.0 > 250)
            w->alive = 0;
    }
    for (int i = 0; i < MAX_DROPS; i++) {
        drop *d = &drops[i];
        if (!d->alive)
            continue;
        active++;
        d->age += dt;
        d->vy += 380 * dt;
        d->x += d->vx * dt;
        d->y += d->vy * dt;
        if (d->y > CY + 120 || d->age > 1.5)
            d->alive = 0;
    }
    for (int i = 0; i < MAX_PUFFS; i++) {
        puff *p = &puffs[i];
        if (!p->alive)
            continue;
        active++;
        p->age += dt;
        p->x += p->vx * dt;
        p->y += p->vy * dt;
        if (p->age >= p->life)
            p->alive = 0;
    }

    /* smooth wobble and heat so stage changes never jump */
    static const double wob_target[5] = { 0, 0, 0.015, 0.045, 0.075 };
    wob_amp += (wob_target[stage] - wob_amp) * fmin(1, dt * 2);
    double heat_target = stage == ST_MELT ? 1 : stage == ST_PANIC ? 0.35 : 0;
    heat += (heat_target - heat) * fmin(1, dt * 1.5);

    if (t > blink_next) {
        blink_start = t;
        blink_next = t + 2.5 + frand() * 3;
    }
    return active;
}

/* ---------------------------------------------------------------- render */

static int mood_now(double t)
{
    double sa = stamp_age(t);
    if (sa >= STAMP_IN && sa < STAMP_HOLD)
        return M_SMUG;
    if (sa >= STAMP_HOLD && sa < STAMP_HOLD + 0.9)
        return M_SHOCK;
    switch (stage) {
    case ST_IDLE:  return M_SLEEP;
    case ST_THINK: return M_THINK;
    case ST_WAIT:  return M_WAIT;
    case ST_PANIC: return M_PANIC;
    default:       return M_MELT;
    }
}

static void eye_white(cairo_t *cr, double x, double y, double rx, double ry)
{
    cairo_save(cr);
    cairo_translate(cr, x, y);
    cairo_scale(cr, rx, ry);
    cairo_arc(cr, 0, 0, 1, 0, 2 * M_PI);
    cairo_restore(cr);
    cairo_set_source_rgb(cr, 1, 1, 1);
    cairo_fill_preserve(cr);
    set_rgb(cr, INK);
    cairo_set_line_width(cr, 2.6);
    cairo_stroke(cr);
}

static void dot(cairo_t *cr, double x, double y, double r, rgb c)
{
    cairo_arc(cr, x, y, r, 0, 2 * M_PI);
    set_rgb(cr, c);
    cairo_fill(cr);
}

static void line(cairo_t *cr, double x0, double y0, double x1, double y1, double w)
{
    cairo_move_to(cr, x0, y0);
    cairo_line_to(cr, x1, y1);
    cairo_set_line_width(cr, w);
    cairo_stroke(cr);
}

/* The face, in sprite-local coordinates (origin at the sprite's centre) */
static void draw_face(cairo_t *cr, int mood, double t)
{
    const double ey = 20, ex = 27, my = 54;
    int blink = t - blink_start < 0.14 && (mood == M_THINK || mood == M_WAIT || mood == M_SMUG);

    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    set_rgb(cr, INK);

    /* eyes */
    if (mood == M_SLEEP || blink) {
        for (int k = -1; k <= 1; k += 2) {
            cairo_new_path(cr);
            cairo_arc_negative(cr, k * ex, ey - 4, 12, M_PI * 0.9, M_PI * 0.1);
            set_rgb(cr, INK);
            cairo_set_line_width(cr, 4);
            cairo_stroke(cr);
        }
    } else if (mood == M_MELT) {
        for (int k = -1; k <= 1; k += 2) {
            eye_white(cr, k * ex, ey, 17, 17);
            cairo_save(cr);
            cairo_translate(cr, k * ex, ey);
            cairo_rotate(cr, k * t * 7);
            cairo_new_path(cr);
            for (double a = 0; a < 4.2 * M_PI; a += 0.25)
                cairo_line_to(cr, cos(a) * a * 1.12, sin(a) * a * 1.12);
            cairo_restore(cr);
            set_rgb(cr, INK);
            cairo_set_line_width(cr, 2.4);
            cairo_stroke(cr);
        }
    } else {
        double rx = 15, ry = 18, pr = 7.5, px = 0, py = 0;
        switch (mood) {
        case M_THINK: px = 5 * sin(t * 0.9); py = -7; break;                  /* eyes up, counting */
        case M_WAIT:  rx = 16; ry = 20; pr = 6; break;
        case M_PANIC: rx = 17; ry = 21; pr = 4.5; px = 6 * sin(t * 3.1); py = 2 * sin(t * 4.3); break;
        case M_SHOCK: rx = 18; ry = 22; pr = 4.5; break;
        case M_SMUG:  px = 4; py = 3; break;
        }
        for (int k = -1; k <= 1; k += 2) {
            eye_white(cr, k * ex, ey, rx, ry);
            dot(cr, k * ex + px, ey + py, pr, INK);
            dot(cr, k * ex + px + pr * 0.35, ey + py - pr * 0.4, pr * 0.3, WHITE);
            if (mood == M_SMUG) {                                            /* heavy eyelids */
                cairo_new_path(cr);
                cairo_rectangle(cr, k * ex - rx - 2, ey - ry - 2, 2 * rx + 4, ry + 3);
                cairo_set_source_rgb(cr, 0.72, 0.06, 0.08);
                cairo_fill(cr);
                set_rgb(cr, INK);
                line(cr, k * ex - rx, ey + 1, k * ex + rx, ey + 1, 3.2);
            }
        }
    }

    /* eyebrows */
    set_rgb(cr, INK);
    switch (mood) {
    case M_THINK:
        line(cr, -ex - 12, ey - 25, -ex + 11, ey - 24, 4);
        line(cr, ex - 11, ey - 31, ex + 12, ey - 36, 4);
        break;
    case M_WAIT:
    case M_SHOCK:
        line(cr, -ex - 12, ey - 33, -ex + 10, ey - 37, 4);
        line(cr, ex - 10, ey - 37, ex + 12, ey - 33, 4);
        break;
    case M_PANIC:
    case M_MELT:
        line(cr, -ex - 12, ey - 26, -ex + 10, ey - 35, 4);
        line(cr, ex - 10, ey - 35, ex + 12, ey - 26, 4);
        break;
    case M_SMUG:
        line(cr, -ex - 12, ey - 22, -ex + 11, ey - 18, 4);
        line(cr, ex - 11, ey - 18, ex + 12, ey - 22, 4);
        break;
    }

    /* mouth */
    cairo_new_path(cr);
    switch (mood) {
    case M_SLEEP:
        cairo_save(cr);
        cairo_translate(cr, 0, my);
        cairo_scale(cr, 5, 4 + 1.2 * sin(t * 1.6));
        cairo_arc(cr, 0, 0, 1, 0, 2 * M_PI);
        cairo_restore(cr);
        cairo_set_source_rgb(cr, 0.35, 0.02, 0.05);
        cairo_fill(cr);
        break;
    case M_THINK:
        cairo_move_to(cr, -12, my);
        cairo_curve_to(cr, -4, my - 4, 4, my + 3, 12, my - 2);
        set_rgb(cr, INK);
        cairo_set_line_width(cr, 3.6);
        cairo_stroke(cr);
        break;
    case M_WAIT:
    case M_SHOCK: {
        double s = mood == M_SHOCK ? 1.5 : 1;
        cairo_save(cr);
        cairo_translate(cr, 0, my);
        cairo_scale(cr, 7 * s, 9 * s);
        cairo_arc(cr, 0, 0, 1, 0, 2 * M_PI);
        cairo_restore(cr);
        cairo_set_source_rgb(cr, 0.35, 0.02, 0.05);
        cairo_fill_preserve(cr);
        set_rgb(cr, INK);
        cairo_set_line_width(cr, 2.6);
        cairo_stroke(cr);
        break;
    }
    case M_PANIC:
        cairo_rectangle(cr, -19, my - 9, 38, 17);
        cairo_set_source_rgb(cr, 1, 1, 1);
        cairo_fill_preserve(cr);
        set_rgb(cr, INK);
        cairo_set_line_width(cr, 2.6);
        cairo_stroke(cr);
        line(cr, -19, my - 0.5, 19, my - 0.5, 2);
        for (int i = -1; i <= 1; i++)
            line(cr, i * 9.5, my - 9, i * 9.5, my + 8, 2);
        break;
    case M_MELT:
        cairo_save(cr);
        cairo_translate(cr, 0, my + 2);
        cairo_scale(cr, 19, 15 + 2 * sin(t * 9));
        cairo_arc(cr, 0, 0, 1, 0, 2 * M_PI);
        cairo_restore(cr);
        cairo_set_source_rgb(cr, 0.30, 0.01, 0.04);
        cairo_fill_preserve(cr);
        set_rgb(cr, INK);
        cairo_set_line_width(cr, 2.6);
        cairo_stroke(cr);
        cairo_save(cr);
        cairo_translate(cr, 0, my + 10);
        cairo_scale(cr, 10, 6);
        cairo_arc(cr, 0, 0, 1, 0, 2 * M_PI);
        cairo_restore(cr);
        cairo_set_source_rgb(cr, 1.0, 0.45, 0.55);
        cairo_fill(cr);
        break;
    case M_SMUG:
        cairo_move_to(cr, -13, my + 1);
        cairo_curve_to(cr, -4, my + 6, 8, my + 3, 15, my - 7);
        set_rgb(cr, INK);
        cairo_set_line_width(cr, 3.8);
        cairo_stroke(cr);
        break;
    }

    /* anime sweat drop on the forehead, sliding slowly */
    if (mood == M_WAIT || mood == M_PANIC || mood == M_MELT || mood == M_SHOCK) {
        double u = fmod(t * 0.35, 1.0), x = 52, y = -14 + u * 26;
        cairo_new_path(cr);
        cairo_move_to(cr, x, y - 16);
        cairo_curve_to(cr, x + 4, y - 6, x + 9, y, x + 9, y + 5);
        cairo_arc(cr, x, y + 5, 9, 0, M_PI);
        cairo_curve_to(cr, x - 9, y, x - 4, y - 6, x, y - 16);
        cairo_close_path(cr);
        cairo_set_source_rgba(cr, 0.65, 0.88, 1.0, 0.95 * (1 - u * u));
        cairo_fill_preserve(cr);
        cairo_set_source_rgba(cr, INK.r, INK.g, INK.b, 0.9 * (1 - u * u));
        cairo_set_line_width(cr, 2);
        cairo_stroke(cr);
    }
}

static void render_scene(cairo_t *cr, double t)
{
    int bw = cairo_image_surface_get_width(berry), bh = cairo_image_surface_get_height(berry);

    cairo_set_operator(cr, CAIRO_OPERATOR_SOURCE);
    cairo_set_source_surface(cr, bg_cache, 0, 0);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);

    if (heat > 0.02) {
        double a = heat * (stage == ST_MELT ? 0.75 + 0.25 * sin(t * 5) : 0.8);
        cairo_set_source_surface(cr, glow_cache, 0, 0);
        cairo_paint_with_alpha(cr, a);
    }

    /* words behind the strawberry */
    for (int pass = 0; pass < 2; pass++) {
        if (pass == 1) {
            /* the strawberry: breathing, and shaking with anxiety as it speeds up */
            double wob = wob_amp * sin(t * (stage >= ST_MELT ? 23 : 17));
            double breathe = stage == ST_IDLE ? 0.018 * sin(t * 1.4) : 0.012 * sin(t * 3);
            double sa = stamp_age(t), squash = 0;
            if (sa >= STAMP_IN && sa < STAMP_IN + 0.3)                     /* proud little bounce */
                squash = 0.06 * sin((sa - STAMP_IN) / 0.3 * M_PI);
            cairo_save(cr);
            cairo_translate(cr, CX, CY + bh / 2.0);
            cairo_rotate(cr, wob);
            cairo_scale(cr, 1 - breathe * 0.5 - squash * 0.5, 1 + breathe + squash);
            cairo_translate(cr, 0, -bh / 2.0);
            cairo_set_source_surface(cr, berry, -bw / 2.0, -bh / 2.0);
            cairo_paint(cr);
            if (heat > 0.02) {
                cairo_set_source_surface(cr, berry_hot, -bw / 2.0, -bh / 2.0);
                cairo_paint_with_alpha(cr, heat);
            }
            draw_face(cr, mood_now(t), t);
            cairo_restore(cr);

            /* sweat and steam */
            for (int i = 0; i < MAX_DROPS; i++) {
                const drop *d = &drops[i];
                if (!d->alive)
                    continue;
                double a = clamp01(1.5 - d->age * 1.1);
                cairo_save(cr);
                cairo_translate(cr, d->x, d->y);
                cairo_rotate(cr, atan2(d->vy, d->vx) - M_PI / 2);
                cairo_move_to(cr, 0, 8);
                cairo_arc(cr, 0, -2, 5, 0, M_PI);
                cairo_close_path(cr);
                cairo_restore(cr);
                cairo_set_source_rgba(cr, 0.62, 0.86, 1.0, a);
                cairo_fill(cr);
            }
            for (int i = 0; i < MAX_PUFFS; i++) {
                const puff *p = &puffs[i];
                if (!p->alive)
                    continue;
                double u = p->age / p->life;
                cairo_arc(cr, p->x, p->y, 8 + 20 * u, 0, 2 * M_PI);
                cairo_set_source_rgba(cr, 0.95, 0.92, 0.95, 0.45 * (1 - u) * clamp01(u * 6));
                cairo_fill(cr);
            }
        }

        for (int i = 0; i < MAX_WORDS; i++) {
            const word *w = &words[i];
            if (!w->alive || w->top != pass)
                continue;
            double a = clamp01(w->age / 0.15) * clamp01((w->life - w->age) / 0.5);
            double dx = fabs(w->x - SIZE / 2.0) + w->img->w / 2.0, dy = fabs(w->y - SIZE / 2.0) + w->img->h / 2.0;
            a *= clamp01((242 - sqrt(dx * dx + dy * dy)) / 24);           /* fade before the round edge */
            if (!w->top)
                a *= clamp01((w->y - w->img->h / 2.0 - 92) / 18);          /* and before the question */
            if (a <= 0.01)
                continue;
            double pop = w->age < 0.15 ? 0.6 + 0.4 * sin(w->age / 0.15 * M_PI / 2) : 1;
            if (pop < 1) {
                cairo_save(cr);
                cairo_translate(cr, w->x, w->y);
                cairo_scale(cr, pop, pop);
                paint_label(cr, w->img, 0, 0, a);
                cairo_restore(cr);
            } else {
                paint_label(cr, w->img, w->x, w->y, a);
            }
        }
    }

    /* FINAL ANSWER stamp: slams in, holds, then drops away when it says "Wait," */
    double sa = stamp_age(t) - STAMP_IN;
    if (sa >= 0 && sa < STAMP_END - STAMP_IN) {
        const label *st = &stamp_img[stamp_answer == 3 ? 0 : 1];
        double scale = 1, a = 1, y = CY + 8, rot = -0.14;
        if (sa < 0.14) {
            double u = sa / 0.14;
            scale = 1.9 - 0.9 * u * u;
            a = u;
        }
        double fall = sa - (STAMP_HOLD - STAMP_IN);
        if (fall > 0) {
            y += 420 * fall * fall;
            rot += 0.9 * fall;
            a = clamp01(1 - fall / 0.7);
        }
        cairo_save(cr);
        cairo_translate(cr, CX, y);
        cairo_rotate(cr, rot);
        cairo_scale(cr, scale, scale);
        paint_label(cr, st, 0, 0, a);
        cairo_restore(cr);
    }

    cairo_set_source_surface(cr, hud_cache, 0, HUD_Y);
    cairo_paint(cr);
}

/* ---------------------------------------------------------------- HUD */

typedef struct { double tok, tok0, tok1, total; } shown_t;

static void fmt_count(char *out, size_t n, double v)
{
    if (v >= 1e9)
        snprintf(out, n, "%.2fB", v / 1e9);
    else if (v >= 1e7)
        snprintf(out, n, "%.1fM", v / 1e6);
    else if (v >= 1e6)
        snprintf(out, n, "%.2fM", v / 1e6);
    else if (v >= 1e3)
        snprintf(out, n, "%d,%03d", (int)(v / 1000), (int)fmod(v, 1000));
    else
        snprintf(out, n, "%d", (int)v);
}

static const char *answer_text(void)
{
    static const char *a[5] = { "ANSWER: 3 \xE2\x9C\x93", "ANSWER: 3?", "ANSWER: 3? 2?", "ANSWER: 2", "ANSWER: 2 (100%)" };
    return a[stage];
}

static rgb answer_color(void)
{
    static const rgb c[5] = { { 0.45, 0.95, 0.55 }, { 1.0, 0.88, 0.35 }, { 1.0, 0.64, 0.25 },
                              { 1.0, 0.38, 0.25 }, { 1.0, 0.22, 0.22 } };
    return c[stage];
}

/* Redraws the HUD strip when anything on it changed; returns 1 if it did */
static int update_hud(const stats *s, const shown_t *sh)
{
    static char last[256];
    char key[256], tok[32], t0[16], t1[16], cnt[32], watts[32], tp0[16], tp1[16];
    int idle = stage == ST_IDLE;

    snprintf(tok, sizeof(tok), "%.0f", idle ? 0 : sh->tok);
    snprintf(t0, sizeof(t0), "%.0f", sh->tok0);
    snprintf(t1, sizeof(t1), "%.0f", sh->tok1);
    fmt_count(cnt, sizeof(cnt), sh->total);
    double w = s->power[0] + s->power[1];
    if (w >= 1000)
        snprintf(watts, sizeof(watts), "%.2f kW", w / 1000);
    else
        snprintf(watts, sizeof(watts), "%.0f W", w);
    snprintf(tp0, sizeof(tp0), "%d\xC2\xB0", s->temp[0]);
    snprintf(tp1, sizeof(tp1), "%d\xC2\xB0", s->temp[1]);
    snprintf(key, sizeof(key), "%d|%s|%s|%s|%s|%s|%s|%s", stage, tok, t0, t1, cnt, watts, tp0, tp1);
    if (!strcmp(key, last))
        return 0;
    snprintf(last, sizeof(last), "%s", key);

    cairo_t *cr = cairo_create(hud_cache);
    cairo_set_operator(cr, CAIRO_OPERATOR_CLEAR);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    cairo_translate(cr, 0, -HUD_Y);                 /* draw in screen coordinates */

    /* tok/s: total in the middle, per server either side */
    const double y1 = 326;
    rgb numc = idle ? DIM : WHITE;
    double wn = text_w(cr, tok, 46), wu = text_w(cr, " tok/s", 20);
    double x = SIZE / 2.0 - (wn + wu) / 2;
    text_at(cr, x, y1, 46, numc, tok, 3.5);
    text_at(cr, x + wn, y1 + 8, 20, (rgb){ 0.75, 0.72, 0.85 }, " tok/s", 2.5);
    if (!idle) {
        text_mid(cr, 64, y1 + 2, 24, BLUE, t0, 3);
        text_mid(cr, SIZE - 64, y1 + 2, 24, ORANGE, t1, 3);
    }

    /* the joke counter: every token the servers ever generated went into this */
    const double y2 = 366;
    const char *pre = "thought for ", *post = " tokens";
    double w1 = text_w(cr, pre, 21), w2 = text_w(cr, cnt, 23), w3 = text_w(cr, post, 21);
    x = SIZE / 2.0 - (w1 + w2 + w3) / 2;
    text_at(cr, x, y2, 21, (rgb){ 0.78, 0.74, 0.90 }, pre, 2.5);
    text_at(cr, x + w1, y2, 23, YELLOW, cnt, 3);
    text_at(cr, x + w1 + w2, y2, 21, (rgb){ 0.78, 0.74, 0.90 }, post, 2.5);

    text_mid(cr, SIZE / 2.0, 402, 28, answer_color(), answer_text(), 3.5);

    /* power and temps */
    const double y4 = 437;
    text_mid(cr, SIZE / 2.0, y4, 20, (rgb){ 0.85, 0.82, 0.92 }, watts, 2.5);
    text_mid(cr, SIZE / 2.0 - 92, y4, 20, BLUE, tp0, 2.5);
    text_mid(cr, SIZE / 2.0 + 92, y4, 20, ORANGE, tp1, 2.5);
    cairo_destroy(cr);
    return 1;
}

/* ---------------------------------------------------------------- showcase */

/*
 * A scripted 42 second loop: asleep, a prompt arrives, then each stage in turn with a
 * FINAL ANSWER in the calm stage and one in the meltdown, then back to sleep.
 */
typedef struct { double t, tok; } keyframe;

static const keyframe script[] = {
    {  0.0,    0 }, {  4.5,    0 }, {  5.0,  240 }, { 12.5,  240 },       /* thinking, stamps 3 */
    { 13.5,  650 }, { 18.5,  650 },                                      /* Wait, */
    { 19.5, 1150 }, { 24.5, 1150 },                                      /* BUT WAIT */
    { 25.5, 1900 }, { 36.5, 1900 },                                      /* meltdown, stamps 2 */
    { 38.0,    0 }, { 42.0,    0 },
};
#define SCRIPT_LEN      (sizeof(script) / sizeof(script[0]))
#define SCRIPT_PERIOD   42.0

static void showcase_poll(stats *s, double t)
{
    double lt = fmod(t, SCRIPT_PERIOD), tok = 0;
    for (size_t i = 0; i + 1 < SCRIPT_LEN; i++) {
        const keyframe *a = &script[i], *b = &script[i + 1];
        if (lt >= a->t && lt < b->t) {
            double u = (lt - a->t) / (b->t - a->t);
            u = u * u * (3 - 2 * u);
            tok = a->tok + (b->tok - a->tok) * u;
            break;
        }
    }
    /* force the two FINAL ANSWER beats to land where the script wants them */
    stamp_next = (lt >= 8.0 && lt < 8.5) || (lt >= 30.0 && lt < 30.5) ? 0 : 1e9;

    double wobble = 1 + 0.05 * sin(t * 2.3);
    s->tok_port[0] = tok * 0.48 * wobble;
    s->tok_port[1] = tok * 0.52 / wobble;
    s->tok_s = s->tok_port[0] + s->tok_port[1];
    s->running = tok > 0 ? 1 + (int)(tok / 300) : 0;
    s->power[0] = 28 + fmin(547, tok * 0.28);
    s->power[1] = 33 + fmin(542, tok * 0.30);
    s->temp[0] = 36 + (int)(tok / 90);
    s->temp[1] = 42 + (int)(tok / 60);
}

/* ---------------------------------------------------------------- main */

static void reset_particles(void)
{
    memset(words, 0, sizeof(words));
    memset(drops, 0, sizeof(drops));
    memset(puffs, 0, sizeof(puffs));
    memset(lane_free, 0, sizeof(lane_free));
}

int main(int argc, char **argv)
{
    cairo_surface_t *surf;
    cairo_t *cr;
    tjhandle tj;
    unsigned char *jpeg = NULL;
    stats s = { 0 };
    shown_t sh = { 0 };
    double last, next_poll = 0, next_hud = 0, t0, demo_total = 1284337;
    int fd = -1, bench = 0, active = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--demo"))
            demo = 1;
        else if (!strcmp(argv[i], "--bench"))
            bench = 1;
        else if (!strcmp(argv[i], "--showcase"))
            showcase = demo = 1;
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
        /* each scene runs until a moment worth previewing; the stamp one catches FINAL ANSWER */
        struct { double tok0, tok1, w0, w1; int stamp; const char *png; } scenes[] = {
            { 950, 1010, 540, 555, 0, "butwait_melt.png" },
            { 330, 360, 330, 350, 0, "butwait_wait.png" },
            { 120, 130, 180, 190, 1, "butwait_final.png" },
            { 0, 0, 25, 30, 0, "butwait_idle.png" },
        };
        srand(7);
        for (int k = 0; k < 4; k++) {
            reset_particles();
            stage = ST_IDLE;
            stage_since = -10;
            stamp_start = -100;
            s.tok_port[0] = scenes[k].tok0; s.tok_port[1] = scenes[k].tok1;
            s.tok_s = s.tok_port[0] + s.tok_port[1]; s.running = s.tok_s > 0 ? 4 : 0;
            s.power[0] = scenes[k].w0; s.power[1] = scenes[k].w1;
            s.temp[0] = s.tok_s > 0 ? 58 : 34; s.temp[1] = s.tok_s > 0 ? 71 : 41;
            sh.tok = s.tok_s; sh.tok0 = s.tok_port[0]; sh.tok1 = s.tok_port[1]; sh.total = 48213377;
            int n = 12 * FPS_BUSY;
            size_t len = 0;
            double b0 = 0, t = 0;
            for (int i = 0; i < n; i++) {
                t = i / (double)FPS_BUSY;
                if (i == n - 60)
                    b0 = now_s();
                if (scenes[k].stamp)
                    stamp_next = i == n - 60 - 12 ? 0 : 1e9;         /* land in the hold */
                else
                    stamp_next = 1e9;
                if (i % (FPS_BUSY / 4) == 0)
                    update_hud(&s, &sh);
                active = simulate(&s, s.tok_s, 1.0 / FPS_BUSY, t);
                render_scene(cr, t);
                cairo_surface_flush(surf);
                tj3Compress8(tj, cairo_image_surface_get_data(surf), SIZE, cairo_image_surface_get_stride(surf),
                             SIZE, TJPF_BGRX, &jpeg, &len);
                if (scenes[k].stamp && i == n - 60 + 20)
                    cairo_surface_write_to_png(surf, scenes[k].png);
            }
            printf("%s: %.2f ms/frame, jpeg %zu bytes, %d particles\n", scenes[k].png,
                   (now_s() - b0) * 1000 / 60, len, active);
            if (!scenes[k].stamp)
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
            demo_total += s.tok_s * dt;
            s.tok_total = demo_total;
        } else if (t >= next_poll) {
            next_poll = t + 1.0;
            if (demo) {
                /* scale a copy each second; scaling s itself would compound */
                static stats demo_base;
                demo_poll(&demo_base, t - t0);
                s = demo_base;
                s.tok_port[0] *= 7; s.tok_port[1] *= 7;
                s.tok_s = s.tok_port[0] + s.tok_port[1];
                if (s.tok_s < 150) { s.tok_s = s.tok_port[0] = s.tok_port[1] = 0; s.running = 0; }
            } else {
                gpus_poll(&s);
                vllm_poll(&s, t);
            }
        }
        if (demo && !showcase) {
            demo_total += s.tok_s * dt;
            s.tok_total = demo_total;
        }

        k = dt * 4 > 1 ? 1 : dt * 4;
        sh.tok  += (s.tok_s - sh.tok) * k;
        sh.tok0 += (s.tok_port[0] - sh.tok0) * k;
        sh.tok1 += (s.tok_port[1] - sh.tok1) * k;
        sh.total = s.tok_total;

        active = simulate(&s, sh.tok, dt, t - t0);
        if (t >= next_hud) {
            next_hud = t + 0.25;
            update_hud(&s, &sh);
        }
        render_scene(cr, t - t0);
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

        /* Slow down when it's asleep */
        int idle = !showcase && stage == ST_IDLE;
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
    if (nvml_ok)
        nvmlShutdown();
    if (fd >= 0)
        close(fd);
    return 0;
}
