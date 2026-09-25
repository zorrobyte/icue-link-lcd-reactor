/*
 * kombat: GPU FIGHTER II TURBO HYPER INFERENCE EDITION on the iCUE LINK AIO pump LCD.
 *
 * Your two graphics cards are 16-bit arcade fighters on a datacenter rooftop: fans for
 * eyes, a PCIe-gold-finger grin, a red headband and a 12VHPWR cable for a ponytail.
 * Each vLLM server's tokens are its attacks: one fireball per 32 tokens, a full-on
 * beam past 300 tok/s (the clash point shows who's faster), Super Saiyan hair at 600,
 * the stage shakes and rocks float past 1200 total, lightning past 2000. Every burst of
 * requests is a ROUND; when it ends the GPU that generated more tokens WINS. Hot cards
 * summon the TOASTY llama; the power plug smokes; IT'S OVER 900 WATTS.
 * The stage and the llama were generated with an image model (assets/kombat/).
 * Run with --demo to simulate data, --showcase for a scripted loop, --bench for PNGs.
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
 * The stage art is cached; the HUD is drawn into its own cached layer only when the
 * numbers change (4x per second). Fighters, beams and effects are procedural cairo,
 * drawn only where they are. The frame rate drops to FPS_IDLE while everyone sleeps.
 */
#define FPS_BUSY         20
#define FPS_IDLE         15
#define FPS_SHOWCASE     30
#define FLOOR_Y          392.0
#define FB_Y             (FLOOR_Y - 66)
#define SC               0.85       /* fighter scale */
#define TOK_PER_FIREBALL 32.0       /* one fireball per this many generated tokens */
#define MAX_FB_RATE      7.0        /* fireballs/sec per fighter; past this the arms just blur */
#define BEAM_TOK         300.0      /* per-server tok/s where the fireballs become a beam */
#define SUPER_TOK        600.0      /* per-server tok/s: Super Saiyan GPU */
#define SHAKE_TOK        1200.0     /* total tok/s: the stage shakes and rocks float */
#define ULTRA_TOK        2000.0     /* total tok/s: lightning, ULTRA COMBO */
#define OVER_W           900.0      /* total watts: IT'S OVER 900 WATTS */
#define SMOKE_W          480.0      /* watts per card where its 12VHPWR plug starts smoking */
#define TOASTY_C         72         /* GPU temp that summons the toasty llama */
#define BAR_MAX          1000.0     /* per-server tok/s that fills a bar */
#define ROUND_END_S      2.0        /* seconds of silence that end a round */
#define MAX_FB           64
#define MAX_PART         400
#define MAX_WORDS        6

static const double HOME_X[2] = { 112, 368 };
static const char  *NAME[2]   = { "ZOTAC", "TUF" };
static const rgb    GOLD      = { 1.0, 0.84, 0.22 };
static const rgb    INK       = { 0.07, 0.02, 0.05 };

typedef struct {
    double xoff, vx;            /* knockback, springs back home */
    double hit_age, throw_age;  /* seconds since last hit taken / fireball thrown */
    double fb_acc, beam_hit_acc, smoke_acc, word_t;
    double sleep, super, beam, vic, dizzy;     /* smoothed pose weights 0..1 */
    double fan;
    double hand_x, hand_y;      /* front glove, scene coords, from the last draw */
    double rate;                /* fireballs/sec right now */
    int    beaming, is_super, combo;
    double combo_t;
} fighter;

typedef struct { double x, age; int side, alive; } fireball;

enum { P_SPARK, P_SMOKE, P_ROCK };
typedef struct { double x, y, vx, vy, age, life, size, rot; rgb col; int type, alive; } particle;

typedef struct { double x, y, age, rot; const char *word; int alive; } hitword;

typedef struct { char main[48], sub[64]; double dur; } ann_t;

enum { R_IDLE, R_FIGHT, R_END };

static fighter  F[2];
static fireball fbs[MAX_FB];
static particle parts[MAX_PART];
static hitword  words[MAX_WORDS];
static ann_t    ann_q[8], ann_cur;
static int      ann_n, ann_on;
static double   ann_start;
static int      rstate, round_no, winner;
static double   round_tok[2], quiet_t, end_t, clash_x = SIZE / 2.0;
static double   last_struggle = -1e9, last_over = -1e9, last_ultra = -1e9, last_super[2] = { -1e9, -1e9 };
static double   toasty_start = -1e9, toasty_next;
static double   shake, shake_x, shake_y, rock_acc, bolt_t = -1e9, bolt_next;
static unsigned bolt_seed;
static double   g_flash;            /* hit flash for the fighter being drawn */

static cairo_surface_t *stage_img, *toasty_img, *hud_cache, *fb_sprite[2], *clash_sprite;
static char hud_key[256];

static double frand(void) { return rand() / (double)RAND_MAX; }

static cairo_surface_t *load_asset(const char *name)
{
    char exe[512], path[1024];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n > 0) {
        exe[n] = 0;
        char *slash = strrchr(exe, '/');
        if (slash)
            *slash = 0;
        snprintf(path, sizeof(path), "%s/assets/kombat/%s", exe, name);
        cairo_surface_t *s = cairo_image_surface_create_from_png(path);
        if (cairo_surface_status(s) == CAIRO_STATUS_SUCCESS)
            return s;
        cairo_surface_destroy(s);
    }
    snprintf(path, sizeof(path), "assets/kombat/%s", name);
    cairo_surface_t *s = cairo_image_surface_create_from_png(path);
    if (cairo_surface_status(s) != CAIRO_STATUS_SUCCESS) {
        fprintf(stderr, "kombat: can't load %s (looked next to the binary and in ./assets/kombat)\n", name);
        exit(1);
    }
    return s;
}

/* A soft glowing ball: white core, coloured halo, transparent edge */
static cairo_surface_t *glow_sprite(int size, rgb col)
{
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, size, size);
    cairo_t *cr = cairo_create(s);
    double r = size / 2.0;
    cairo_pattern_t *g = cairo_pattern_create_radial(r, r, 0, r, r, r);
    cairo_pattern_add_color_stop_rgba(g, 0.00, 1, 1, 1, 1);
    cairo_pattern_add_color_stop_rgba(g, 0.22, 1, 1, 1, 1);
    cairo_pattern_add_color_stop_rgba(g, 0.40, col.r, col.g, col.b, 0.95);
    cairo_pattern_add_color_stop_rgba(g, 0.70, col.r, col.g, col.b, 0.30);
    cairo_pattern_add_color_stop_rgba(g, 1.00, col.r, col.g, col.b, 0);
    cairo_set_source(cr, g);
    cairo_paint(cr);
    cairo_pattern_destroy(g);
    cairo_destroy(cr);
    return s;
}

static void load_assets(void)
{
    stage_img = load_asset("stage.png");
    toasty_img = load_asset("toasty.png");
    fb_sprite[0] = glow_sprite(64, BLUE);
    fb_sprite[1] = glow_sprite(64, ORANGE);
    clash_sprite = glow_sprite(128, (rgb){ 1.0, 0.85, 0.35 });
    hud_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, SIZE);
}

static void reset_scene(void)
{
    memset(F, 0, sizeof(F));
    memset(fbs, 0, sizeof(fbs));
    memset(parts, 0, sizeof(parts));
    memset(words, 0, sizeof(words));
    for (int i = 0; i < 2; i++)
        F[i].sleep = 1, F[i].hit_age = F[i].throw_age = 10, F[i].combo_t = -1e9;
    ann_n = ann_on = 0;
    rstate = R_IDLE;
    round_no = 0;
    quiet_t = 100;
    clash_x = SIZE / 2.0;
    toasty_start = -1e9;
    toasty_next = 0;
    last_struggle = last_over = last_ultra = last_super[0] = last_super[1] = -1e9;
    shake = 0;
    bolt_t = -1e9;
    bolt_next = 0;
    hud_key[0] = 0;
}

/* ---------------------------------------------------------------- simulation */

static void announce(const char *main, const char *sub, double dur, int force)
{
    if ((!force && (ann_n >= 1 || (ann_on && ann_cur.dur > 1.3))) || ann_n >= 8)
        return;
    ann_t *a = &ann_q[ann_n++];
    snprintf(a->main, sizeof(a->main), "%s", main);
    snprintf(a->sub, sizeof(a->sub), "%s", sub ? sub : "");
    a->dur = dur;
}

static void ann_update(double t)
{
    if (ann_on && t - ann_start < ann_cur.dur)
        return;
    ann_on = 0;
    if (ann_n > 0) {
        ann_cur = ann_q[0];
        memmove(ann_q, ann_q + 1, (size_t)(--ann_n) * sizeof(ann_t));
        ann_on = 1;
        ann_start = t;
    }
}

static particle *new_part(int type)
{
    for (int i = 0; i < MAX_PART; i++)
        if (!parts[i].alive) {
            memset(&parts[i], 0, sizeof(particle));
            parts[i].alive = 1;
            parts[i].type = type;
            return &parts[i];
        }
    return NULL;
}

static void sparks(double x, double y, int n, rgb col, double speed)
{
    for (int k = 0; k < n; k++) {
        particle *p = new_part(P_SPARK);
        if (!p)
            return;
        double a = frand() * 2 * M_PI, v = speed * (0.4 + frand() * 0.8);
        p->x = x;
        p->y = y;
        p->vx = cos(a) * v;
        p->vy = sin(a) * v - speed * 0.3;
        p->life = 0.25 + frand() * 0.3;
        p->col = frand() < 0.4 ? (rgb){ 1, 1, 0.85 } : col;
    }
}

static void pop_word(double x, double y)
{
    static const char *list[] = { "POW!", "BAM!", "WHAM!", "KAPOW!", "BONK!", "OOF!", "ZAP!", "YEET!", "SMAK!", "404!" };
    for (int i = 0; i < MAX_WORDS; i++)
        if (!words[i].alive) {
            words[i] = (hitword){ x, y, 0, (frand() - 0.5) * 0.5, list[rand() % 10], 1 };
            return;
        }
}

static const rgb *team(int i) { return i == 0 ? &BLUE : &ORANGE; }

/* victim takes a hit from attacker at (x, y) */
static void hit(int victim, int attacker, double push, double x, double y, double t)
{
    fighter *v = &F[victim], *a = &F[attacker];
    v->hit_age = 0;
    v->vx += (victim == 0 ? -1 : 1) * push;
    if (v->combo >= 15) {
        char sub[64];
        snprintf(sub, sizeof(sub), "%s's %d HITS, GONE", NAME[victim], v->combo);
        announce("C-C-C-COMBO BREAKER!", sub, 1.6, 0);
    }
    v->combo = 0;
    if (t - a->combo_t > 2.0)
        a->combo = 0;
    a->combo++;
    a->combo_t = t;
    static const int milestones[] = { 25, 50, 100, 250, 500, 1000, 9001 };
    for (size_t k = 0; k < sizeof(milestones) / sizeof(milestones[0]); k++)
        if (a->combo == milestones[k]) {
            char m[48];
            snprintf(m, sizeof(m), "%d HIT COMBO!!", a->combo);
            announce(m, NAME[attacker], 1.4, 0);
        }
    sparks(x, y, 8, *team(attacker), 260);
    if (t - a->word_t > 0.7) {
        a->word_t = t;
        pop_word(x + (frand() - 0.5) * 30, y - 34 - frand() * 20);
    }
    shake = fmax(shake, 3);
}

static double fighter_x(int i) { return HOME_X[i] + F[i].xoff; }

static void start_round(double t)
{
    char m[32];
    (void)t;
    round_no++;
    round_tok[0] = round_tok[1] = 0;
    rstate = R_FIGHT;
    ann_n = 0;
    ann_on = 0;
    snprintf(m, sizeof(m), "ROUND %d", round_no);
    announce(m, NULL, 1.1, 1);
    announce("FIGHT!", NULL, 0.9, 1);
}

static void end_round(double t)
{
    char m[48], sub[64];
    double a = round_tok[0], b = round_tok[1];
    rstate = R_END;
    end_t = t;
    ann_n = 0;
    ann_on = 0;
    if (a + b < 1 || fabs(a - b) < 0.04 * fmax(a, b))
        winner = -1;
    else
        winner = a > b ? 0 : 1;
    announce("K.O.!", NULL, 1.0, 1);
    if (gpu_source)                         /* round totals as GPU-seconds of full activity */
        snprintf(sub, sizeof(sub), "%.0f vs %.0f GPU-SECONDS", a / GPU_FULL_RATE, b / GPU_FULL_RATE);
    else
        snprintf(sub, sizeof(sub), "%.0f vs %.0f TOKENS", a, b);
    if (winner < 0) {
        announce("DOUBLE K.O.", sub, 2.6, 1);
    } else {
        snprintf(m, sizeof(m), "%s WINS", NAME[winner]);
        announce(m, sub, 1.8, 1);
        if (round_tok[1 - winner] < 1)
            announce("FLAWLESS VICTORY", NULL, 1.6, 1);
    }
}

static void simulate(const stats *s, double dt, double t)
{
    int busy = s->tok_s >= 1 || s->running > 0;

    /* rounds: a burst of requests is a round, silence ends it */
    if (busy) {
        quiet_t = 0;
        if (rstate != R_FIGHT)
            start_round(t);
    } else {
        quiet_t += dt;
        if (rstate == R_FIGHT && quiet_t > ROUND_END_S)
            end_round(t);
    }
    if (rstate == R_END && t - end_t > 7)
        rstate = R_IDLE;
    if (rstate == R_FIGHT)
        for (int i = 0; i < 2; i++)
            round_tok[i] += s->tok_port[i] * dt;

    /* fighters */
    for (int i = 0; i < 2; i++) {
        fighter *f = &F[i];
        double tok = s->tok_port[i];
        int awake = tok >= 1 || s->running_port[i] > 0;
        if (f->beaming ? tok < BEAM_TOK * 0.8 : tok >= BEAM_TOK)
            f->beaming = !f->beaming;
        if (f->is_super ? tok < SUPER_TOK * 0.9 : tok >= SUPER_TOK) {
            f->is_super = !f->is_super;
            if (f->is_super && t - last_super[i] > 30) {
                last_super[i] = t;
                announce("SUPER SAIYAN GPU!", NAME[i], 1.6, 0);
            }
        }
        /* a quiet server naps mid-round; if both go quiet they hold their stance until the K.O. */
        double w_sleep = rstate == R_IDLE || (rstate == R_FIGHT && !awake && busy) ? 1 : 0;
        double w_vic = rstate == R_END && winner == i;
        double w_dizzy = rstate == R_END && winner != i;
        double k = fmin(1, dt * 6);
        f->sleep += (w_sleep - f->sleep) * k;
        f->vic += (w_vic - f->vic) * k;
        f->dizzy += (w_dizzy - f->dizzy) * k;
        f->beam += ((f->beaming && rstate == R_FIGHT) - f->beam) * k;
        f->super += (f->is_super - f->super) * fmin(1, dt * 3);
        f->fan += dt * (1 + 45 * s->load[i]) * (1 - 0.9 * f->sleep);
        f->hit_age += dt;
        f->throw_age += dt;

        /* knockback spring */
        f->vx += (-45 * f->xoff - 9 * f->vx) * dt;
        f->xoff += f->vx * dt;
        if (fabs(f->xoff) > 60)
            f->xoff = copysign(60, f->xoff), f->vx = 0;

        /* fireballs: one per TOK_PER_FIREBALL tokens */
        f->rate = 0;
        if (!f->beaming && awake && rstate == R_FIGHT) {
            f->rate = fmin(tok / TOK_PER_FIREBALL, MAX_FB_RATE);
            f->fb_acc += f->rate * dt;
            while (f->fb_acc >= 1) {
                f->fb_acc -= 1;
                f->throw_age = 0;
                for (int j = 0; j < MAX_FB; j++)
                    if (!fbs[j].alive) {
                        fbs[j] = (fireball){ fighter_x(i) + (i == 0 ? 90 : -90) * SC, 0, i, 1 };
                        break;
                    }
            }
        } else {
            f->fb_acc = 0;
        }

        /* the power plug smokes when the card pulls a lot */
        if (s->power[i] > SMOKE_W) {
            f->smoke_acc += dt * (s->power[i] - SMOKE_W) / 22;
            while (f->smoke_acc >= 1) {
                f->smoke_acc -= 1;
                particle *p = new_part(P_SMOKE);
                if (p) {
                    p->x = fighter_x(i) + (i == 0 ? -31 : 31) + (frand() - 0.5) * 6;
                    p->y = FLOOR_Y - 112 + f->sleep * 17;
                    p->vx = (i == 0 ? -12 : 12) + (frand() - 0.5) * 10;
                    p->vy = -30 - frand() * 20;
                    p->life = 1.4 + frand() * 0.8;
                    p->size = 5 + frand() * 4;
                }
            }
        }
    }

    /* beams */
    int b0 = F[0].beaming && rstate == R_FIGHT, b1 = F[1].beaming && rstate == R_FIGHT;
    if (b0 && b1) {
        double a = s->tok_port[0], b = s->tok_port[1];
        double target = SIZE / 2.0 + 120 * (a - b) / fmax(1, a + b);
        target = fmax(fighter_x(0) + 110, fmin(fighter_x(1) - 110, target));
        clash_x += (target - clash_x) * fmin(1, dt * 2.5);
        if (frand() < dt * 30)
            sparks(clash_x, FB_Y, 2, frand() < 0.5 ? BLUE : ORANGE, 300);
        shake = fmax(shake, 1.2);
        if (t - last_struggle > 25) {
            last_struggle = t;
            announce("BEAM STRUGGLE!!", "FASTER GPU PUSHES THE CLASH", 1.6, 0);
        }
    } else {
        clash_x += (SIZE / 2.0 - clash_x) * fmin(1, dt * 2);
        for (int i = 0; i < 2; i++) {
            int o = 1 - i;
            if (!(i == 0 ? b0 : b1))
                continue;
            fighter *f = &F[i];
            f->beam_hit_acc += dt;
            if (f->beam_hit_acc > 0.15) {
                f->beam_hit_acc = 0;
                hit(o, i, 55, fighter_x(o) + (o == 0 ? 44 : -44), FB_Y, t);
            }
        }
    }

    /* fireballs fly, clash in mid-air, get eaten by beams, or land */
    for (int j = 0; j < MAX_FB; j++) {
        fireball *fb = &fbs[j];
        if (!fb->alive)
            continue;
        int i = fb->side, o = 1 - i;
        double d = i == 0 ? 1 : -1;
        fb->age += dt;
        fb->x += d * 290 * dt;
        for (int m = 0; m < MAX_FB; m++) {
            fireball *e = &fbs[m];
            if (e->alive && e->side != i && fabs(e->x - fb->x) < 22) {
                sparks((fb->x + e->x) / 2, FB_Y, 10, frand() < 0.5 ? BLUE : ORANGE, 240);
                fb->alive = e->alive = 0;
                shake = fmax(shake, 1.5);
                break;
            }
        }
        if (!fb->alive)
            continue;
        if (o == 0 ? b0 : b1) {
            /* the opponent's beam already reaches our fighter: fireballs fizzle in it */
            sparks(fb->x, FB_Y, 6, *team(i), 200);
            fb->alive = 0;
            continue;
        }
        double body = fighter_x(o) - d * 44;
        if ((d > 0 && fb->x >= body) || (d < 0 && fb->x <= body)) {
            hit(o, i, 70, body, FB_Y, t);
            fb->alive = 0;
        }
    }

    /* the stage itself reacts to total throughput */
    shake *= exp(-dt * 8);
    if (s->tok_s >= SHAKE_TOK && rstate == R_FIGHT) {
        shake = fmax(shake, fmin(4.5, 1.5 + (s->tok_s - SHAKE_TOK) / 350));
        rock_acc += dt * fmin(9, 2 + (s->tok_s - SHAKE_TOK) / 150);
        while (rock_acc >= 1) {
            rock_acc -= 1;
            particle *p = new_part(P_ROCK);
            if (p) {
                p->x = 50 + frand() * 380;
                p->y = FLOOR_Y + 2 + frand() * 18;
                p->vy = -18 - frand() * 30;
                p->vx = (frand() - 0.5) * 8;
                p->life = 5 + frand() * 3;
                p->size = 4 + frand() * 9;
                p->rot = frand() * 6;
            }
        }
    }
    shake_x = (frand() - 0.5) * 2 * shake;
    shake_y = (frand() - 0.5) * 2 * shake;
    if (s->tok_s >= ULTRA_TOK && rstate == R_FIGHT) {
        if (t >= bolt_next) {
            bolt_t = t;
            bolt_seed = (unsigned)rand();
            bolt_next = t + 0.35 + frand() * 0.7;
        }
        if (t - last_ultra > 20) {
            char sub[48];
            last_ultra = t;
            snprintf(sub, sizeof(sub), "%.0f%s", shown_rate(s->tok_s), gpu_source ? "% GPU" : " TOK/S");
            announce("ULTRA COMBO!!!", sub, 1.6, 0);
        }
    }

    double watts = s->power[0] + s->power[1];
    if (watts > OVER_W && t - last_over > 30) {
        char sub[48];
        last_over = t;
        snprintf(sub, sizeof(sub), "POWER LEVEL: %.0f W", watts);
        announce("IT'S OVER 900 WATTS!!!", sub, 1.8, 0);
    }

    int hot = s->temp[0] >= TOASTY_C || s->temp[1] >= TOASTY_C;
    if (hot && t >= toasty_next) {
        toasty_start = t;
        toasty_next = t + 40;
    } else if (!hot && toasty_next > t + 10) {
        toasty_next = t + 10;         /* cooled down: allowed back soon after it heats up */
    }

    for (int i = 0; i < MAX_PART; i++) {
        particle *p = &parts[i];
        if (!p->alive)
            continue;
        p->age += dt;
        if (p->age >= p->life) {
            p->alive = 0;
            continue;
        }
        if (p->type == P_SPARK)
            p->vy += 420 * dt;
        else if (p->type == P_ROCK)
            p->rot += dt * 1.5, p->vx += sin(t * 2 + i) * dt * 6;
        p->x += p->vx * dt;
        p->y += p->vy * dt;
    }
    for (int i = 0; i < MAX_WORDS; i++)
        if (words[i].alive && (words[i].age += dt) > 0.7)
            words[i].alive = 0;

    ann_update(t);
}

/* ---------------------------------------------------------------- render */

static void text_path(cairo_t *cr, const char *font, int bold, double x, double y, double size,
                      double max_w, int align, const char *s, double *out_size)
{
    cairo_text_extents_t ext;
    cairo_select_font_face(cr, font, CAIRO_FONT_SLANT_NORMAL, bold ? CAIRO_FONT_WEIGHT_BOLD : CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, size);
    cairo_text_extents(cr, s, &ext);
    if (max_w > 0 && ext.width > max_w) {
        size *= max_w / ext.width;
        cairo_set_font_size(cr, size);
        cairo_text_extents(cr, s, &ext);
    }
    double x0 = align < 0 ? x - ext.x_bearing : align > 0 ? x - ext.width - ext.x_bearing
                                                          : x - ext.width / 2 - ext.x_bearing;
    cairo_new_path(cr);
    cairo_move_to(cr, x0, y - ext.height / 2 - ext.y_bearing);
    cairo_text_path(cr, s);
    if (out_size)
        *out_size = size;
}

/* Usable width of the round screen for text centered at y with height h */
static double chord_width(double y, double h)
{
    double c = SIZE / 2.0, r = SIZE / 2.0 - 16;
    double dy = fmax(fabs(y - h / 2 - c), fabs(y + h / 2 - c));
    return dy >= r ? 0 : 2 * sqrt(r * r - dy * dy);
}

/* Arcade text: thick dark outline, solid fill */
static void arcade_text(cairo_t *cr, const char *font, int bold, double x, double y, double size, double max_w,
                        int align, rgb fill, const char *s)
{
    double sz;
    if (align == 0 && fabs(x - SIZE / 2.0) < 1) {
        double fit = chord_width(y, size * 0.8) - size * 0.25;
        if (fit > 0 && (max_w <= 0 || fit < max_w))
            max_w = fit;
    }
    text_path(cr, font, bold, x, y, size, max_w, align, s, &sz);
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
    cairo_set_line_width(cr, fmax(3, sz * 0.16));
    set_rgb(cr, INK);
    cairo_stroke_preserve(cr);
    set_rgb(cr, fill);
    cairo_fill(cr);
}

static void fset(cairo_t *cr, rgb c) { set_rgb(cr, lerp(c, (rgb){ 1, 1, 1 }, g_flash)); }

static void limb(cairo_t *cr, double x0, double y0, double x1, double y1, double bend, double w, rgb c)
{
    double mx = (x0 + x1) / 2, my = (y0 + y1) / 2, dx = x1 - x0, dy = y1 - y0, l = hypot(dx, dy) + 1e-6;
    double ex = mx - dy / l * bend, ey = my + dx / l * bend;
    cairo_new_path(cr);
    cairo_move_to(cr, x0, y0);
    cairo_line_to(cr, ex, ey);
    cairo_line_to(cr, x1, y1);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
    cairo_set_line_width(cr, w + 4);
    fset(cr, INK);
    cairo_stroke_preserve(cr);
    cairo_set_line_width(cr, w);
    fset(cr, c);
    cairo_stroke(cr);
}

static void glove(cairo_t *cr, double x, double y, rgb c, double alpha)
{
    cairo_new_path(cr);
    cairo_arc(cr, x, y, 14, 0, 2 * M_PI);
    cairo_set_source_rgba(cr, INK.r, INK.g, INK.b, alpha);
    cairo_set_line_width(cr, 3.5);
    cairo_stroke_preserve(cr);
    rgb g = lerp(c, (rgb){ 1, 1, 1 }, g_flash);
    cairo_set_source_rgba(cr, g.r, g.g, g.b, alpha);
    cairo_fill(cr);
    cairo_arc(cr, x + 3, y - 5, 4, 0, 2 * M_PI);
    cairo_set_source_rgba(cr, 1, 1, 1, 0.6 * alpha);
    cairo_fill(cr);
}

static void rounded_rect(cairo_t *cr, double x, double y, double w, double h, double r)
{
    cairo_new_path(cr);
    cairo_arc(cr, x + w - r, y + r, r, -M_PI / 2, 0);
    cairo_arc(cr, x + w - r, y + h - r, r, 0, M_PI / 2);
    cairo_arc(cr, x + r, y + h - r, r, M_PI / 2, M_PI);
    cairo_arc(cr, x + r, y + r, r, M_PI, 3 * M_PI / 2);
    cairo_close_path(cr);
}

static void star_path(cairo_t *cr, double x, double y, double r)
{
    cairo_new_path(cr);
    for (int k = 0; k < 10; k++) {
        double a = -M_PI / 2 + k * M_PI / 5, rr = k & 1 ? r * 0.45 : r;
        cairo_line_to(cr, x + cos(a) * rr, y + sin(a) * rr);
    }
    cairo_close_path(cr);
}

/*
 * One fighter, facing the middle. Local coords: origin at the feet, +x toward the
 * opponent (the right fighter is drawn mirrored), body is a graphics card.
 */
static void draw_fighter(cairo_t *cr, int i, double t)
{
    fighter *f = &F[i];
    const rgb tc = *team(i), BODY = { 0.33, 0.34, 0.40 }, LIMB = { 0.13, 0.13, 0.17 };
    double d = i == 0 ? 1 : -1;
    double awake = 1 - f->sleep;
    double throw_env = f->throw_age < 0.07 ? f->throw_age / 0.07 : exp(-(f->throw_age - 0.07) * 9);
    if (f->rate > 3.5)
        throw_env = fmax(throw_env, 0.55);     /* arms stay out when it's raining fireballs */
    double hit_env = exp(-f->hit_age * 7);
    double reach = fmax(throw_env, f->beam) * awake;
    double bob = awake * sin(t * 5.5 + i * 1.3) * 3 + f->sleep * sin(t * 1.6 + i) * 1.5;
    double jump = -fabs(sin(t * 6.5)) * 28 * f->vic;
    double hipY = -42 + 20 * f->sleep + bob * 0.5;
    double tilt = 0.12 * reach - 0.35 * hit_env * awake - 0.20 * f->sleep + sin(t * 3.5) * 0.14 * f->dizzy;
    double jx = f->hit_age < 0.2 ? (frand() - 0.5) * 6 : 0;
    double fx = fighter_x(i);

    g_flash = f->hit_age < 0.05 ? 0.7 : 0;

    /* shadow */
    cairo_save(cr);
    cairo_translate(cr, fx, FLOOR_Y + 2);
    cairo_scale(cr, 1, 0.22);
    cairo_arc(cr, 0, 0, 50 + jump * 0.4, 0, 2 * M_PI);
    cairo_restore(cr);
    cairo_set_source_rgba(cr, 0, 0, 0.05, 0.45);
    cairo_fill(cr);

    cairo_save(cr);
    cairo_translate(cr, fx + jx, FLOOR_Y + jump);
    cairo_scale(cr, d * SC, SC);

    /* legs */
    for (int k = -1; k <= 1; k += 2) {
        double hx = 15 * k, kx = 32 * k, fxx = 34 * k;
        limb(cr, hx, hipY, fxx, -4, -k * 10 * (1 - f->sleep * 0.5), 11, LIMB);
        cairo_save(cr);
        cairo_translate(cr, fxx + 7, -5);
        cairo_scale(cr, 1.7, 0.75);
        cairo_new_path(cr);
        cairo_arc(cr, 0, 0, 8, 0, 2 * M_PI);
        cairo_restore(cr);
        fset(cr, INK);
        cairo_set_line_width(cr, 3);
        cairo_stroke_preserve(cr);
        fset(cr, (rgb){ 0.93, 0.93, 0.95 });
        cairo_fill(cr);
        (void)kx;
    }

    /* body frame: card centre, tilted about the hips */
    cairo_translate(cr, 0, hipY);
    cairo_rotate(cr, tilt);
    cairo_translate(cr, 0, -40);

    /* Super Saiyan aura: flickering flame tongues licking upward */
    if (f->super > 0.02) {
        for (int layer = 0; layer < 2; layer++) {
            double sc = layer ? 0.78 : 1.0;
            cairo_new_path(cr);
            int n = 26;
            for (int k = 0; k < n; k++) {
                double a = k * 2 * M_PI / n, up = fmax(0, -sin(a));
                double r = k & 1 ? 62 : 74 + 14 * sin(t * 19 + k * 2.7 + layer);
                if (!(k & 1))
                    r += 70 * up * up * (0.75 + 0.25 * sin(t * 13 + k * 1.3));
                double sway = up * 12 * sin(t * 8 + k * 0.7);
                cairo_line_to(cr, cos(a) * r * 0.95 * sc + sway, (sin(a) * r * sc) + 18);
            }
            cairo_close_path(cr);
            cairo_set_source_rgba(cr, 1, layer ? 0.97 : 0.85, layer ? 0.65 : 0.25, (layer ? 0.28 : 0.22) * f->super);
            cairo_fill_preserve(cr);
            cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
            cairo_set_line_width(cr, layer ? 2.5 : 4);
            cairo_set_source_rgba(cr, 1, 0.93, 0.45, (layer ? 0.6 : 0.9) * f->super);
            cairo_stroke(cr);
        }
    }

    /* 12VHPWR cable: the ponytail, trailing to the floor behind */
    cairo_new_path(cr);
    cairo_move_to(cr, -37, -44);
    cairo_curve_to(cr, -55, -85, -100, -50, -78, 40 - hipY);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    cairo_set_line_width(cr, 8);
    fset(cr, (rgb){ 0.04, 0.04, 0.05 });
    cairo_stroke_preserve(cr);
    cairo_set_line_width(cr, 2);
    cairo_set_source_rgba(cr, 1, 1, 1, 0.18);
    cairo_stroke(cr);

    rgb gb[5] = { { 44, 0, 0 }, { 92, 14, 0 }, { -52, 22, 0 }, { -30, 42, 0 }, { -44, 34, 0 } };
    rgb gf[5] = { { 62, -14, 0 }, { 100, 4, 0 }, { 40, -92, 0 }, { 36, 44, 0 }, { 54, 30, 0 } };
    double pose[2][2];
    for (int a = 0; a < 2; a++) {
        rgb *g = a ? gf : gb;
        double x = g[0].r + (g[1].r - g[0].r) * reach, y = g[0].g + (g[1].g - g[0].g) * reach;
        x += (g[2].r - x) * f->vic;
        y += (g[2].g - y) * f->vic;
        if (a)
            y -= f->vic * fabs(sin(t * 6.5)) * 16;     /* fist pump */
        x += (g[4].r - x) * f->dizzy + f->dizzy * sin(t * 5 + a) * 8;
        y += (g[4].g - y) * f->dizzy;
        x += (g[3].r - x) * f->sleep;
        y += (g[3].g - y) * f->sleep;
        if (reach < 0.3 && awake > 0.5)
            y += sin(t * 5.5 + a) * 3;
        pose[a][0] = x;
        pose[a][1] = y;
    }
    limb(cr, -24, 18, pose[0][0], pose[0][1], 12, 9, LIMB);

    /* the card */
    double rx = -56, ry = -40, rw = 112, rh = 80;
    rounded_rect(cr, rx, ry, rw, rh, 10);
    fset(cr, BODY);
    cairo_fill_preserve(cr);
    cairo_set_line_width(cr, 3.5);
    fset(cr, INK);
    cairo_stroke(cr);
    /* shroud highlight + RGB strip down the back edge */
    cairo_rectangle(cr, -50, -28, 3, 58);
    fset(cr, tc);
    cairo_fill(cr);

    /* fans = eyes */
    for (int e = 0; e < 2; e++) {
        double ex = e ? 27 : -21, ey = 0;
        cairo_new_path(cr);
        cairo_arc(cr, ex, ey, 20, 0, 2 * M_PI);
        fset(cr, (rgb){ 0.05, 0.05, 0.07 });
        cairo_fill_preserve(cr);
        cairo_set_line_width(cr, 3.5);
        fset(cr, tc);
        cairo_stroke(cr);
        if (f->sleep < 0.6) {
            double spin = f->fan * (e ? 1 : -1) * (1 + f->dizzy * 3);
            for (int b = 0; b < 7; b++) {
                double a = spin + b * 2 * M_PI / 7;
                cairo_new_path(cr);
                cairo_move_to(cr, ex, ey);
                cairo_arc(cr, ex, ey, 16.5, a, a + 0.55);
                cairo_close_path(cr);
            }
            fset(cr, (rgb){ 0.62, 0.64, 0.70 });
            cairo_fill(cr);
            cairo_new_path(cr);
            cairo_arc(cr, ex, ey, 7, 0, 2 * M_PI);
            fset(cr, f->super > 0.5 ? (rgb){ 1, 0.95, 0.6 } : tc);
            cairo_fill(cr);
            cairo_arc(cr, ex + 3, ey - 3, 2.5, 0, 2 * M_PI);
            cairo_set_source_rgb(cr, 1, 1, 1);
            cairo_fill(cr);
        } else {
            /* eyelid */
            cairo_new_path(cr);
            cairo_arc(cr, ex, ey, 18.5, 0, 2 * M_PI);
            fset(cr, lerp(BODY, INK, 0.2));
            cairo_fill(cr);
            cairo_new_path(cr);
            cairo_arc_negative(cr, ex, ey - 4, 12, M_PI * 0.9, M_PI * 0.1);
            cairo_set_line_width(cr, 3.5);
            fset(cr, INK);
            cairo_stroke(cr);
        }
        /* eyebrows: angry toward the opponent, raised when hit, flat when asleep */
        double lift = 6 * hit_env * awake, angry = 7 * awake * (1 - hit_env);
        cairo_new_path(cr);
        cairo_move_to(cr, ex - 15, -26 - lift - angry * 0.4);
        cairo_line_to(cr, ex + 14, -26 - lift + angry * 0.6);
        cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
        cairo_set_line_width(cr, 6);
        fset(cr, INK);
        cairo_stroke(cr);
    }

    /* PCIe gold-finger grin */
    double shout = fmax(reach, fmax(f->vic, hit_env * awake));
    double mh = 6 + 10 * shout, my = 24;
    cairo_rectangle(cr, 2, my, 46, mh);
    fset(cr, (rgb){ 0.30, 0.02, 0.05 });
    cairo_fill_preserve(cr);
    cairo_set_line_width(cr, 2.5);
    fset(cr, INK);
    cairo_stroke(cr);
    for (int k = 0; k < 8; k++)
        cairo_rectangle(cr, 4 + k * 5.5, my, 3.5, 4 + 1.5 * shout);
    fset(cr, (rgb){ 1.0, 0.78, 0.25 });
    cairo_fill(cr);

    /* Super Saiyan hair: big golden spikes swept back */
    if (f->super > 0.02) {
        static const double hx[5] = { 44, 22, 0, -22, -44 }, hh[5] = { 40, 58, 50, 62, 44 };
        cairo_new_path(cr);
        cairo_move_to(cr, 56, -36);
        for (int k = 0; k < 5; k++) {
            double h = hh[k] * f->super + 3 * sin(t * 18 + k);
            cairo_line_to(cr, hx[k] - 16, -38 - h);
            cairo_line_to(cr, hx[k] - 14, -38);
        }
        cairo_line_to(cr, -80, -44 - 24 * f->super);
        cairo_line_to(cr, -56, -30);
        cairo_close_path(cr);
        cairo_set_line_join(cr, CAIRO_LINE_JOIN_MITER);
        cairo_set_line_width(cr, 3.5);
        cairo_set_source_rgba(cr, 0.45, 0.22, 0.02, f->super);
        cairo_stroke_preserve(cr);
        cairo_set_source_rgba(cr, 1.0, 0.90, 0.30, f->super);
        cairo_fill(cr);
    }

    /* headband with flapping tails */
    cairo_rectangle(cr, -57, -41, 114, 10);
    fset(cr, (rgb){ 0.86, 0.10, 0.13 });
    cairo_fill(cr);
    for (int k = 0; k < 2; k++) {
        cairo_new_path(cr);
        cairo_move_to(cr, -56, -36);
        double flap = 8 + 10 * f->beam + 8 * f->super;
        for (int s2 = 1; s2 <= 6; s2++)
            cairo_line_to(cr, -56 - s2 * 8, -36 + s2 * (2.5 + k * 1.8) * (1 - f->beam * 0.6) +
                                                 sin(t * flap + s2 * 0.9 + k * 1.7) * s2 * 1.1);
        cairo_set_line_width(cr, 6);
        cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
        cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
        fset(cr, (rgb){ 0.86, 0.10, 0.13 });
        cairo_stroke(cr);
    }
    /* 12VHPWR plug */
    cairo_rectangle(cr, -46, -52, 18, 12);
    fset(cr, (rgb){ 0.08, 0.08, 0.09 });
    cairo_fill_preserve(cr);
    cairo_set_line_width(cr, 2);
    fset(cr, (rgb){ 0.9, 0.75, 0.2 });
    cairo_stroke(cr);

    /* front arm, with ghost arms when throwing flat out */
    if (f->rate > 3.5 && awake > 0.5) {
        for (int gh = 1; gh <= 2; gh++) {
            double u = 0.5 + 0.5 * sin(t * 40 + gh * 2.1);
            double gx = gf[0].r + (gf[1].r - gf[0].r) * u, gy = gf[0].g + (gf[1].g - gf[0].g) * u + gh * 10 - 12;
            glove(cr, gx, gy, tc, 0.35);
        }
    }
    limb(cr, 28, 18, pose[1][0], pose[1][1], 12, 9, LIMB);
    glove(cr, pose[1][0], pose[1][1], tc, 1);
    cairo_user_to_device(cr, &pose[1][0], &pose[1][1]);
    f->hand_x = pose[1][0] - shake_x;
    f->hand_y = pose[1][1] - shake_y;

    cairo_restore(cr);
    g_flash = 0;

    /* over the head: Zzz or dizzy stars */
    double headx = fx + d * 6, heady = FLOOR_Y + (hipY - 40 - 58) * SC + jump;
    if (f->sleep > 0.5) {
        for (int k = 0; k < 3; k++) {
            double u = fmod(t * 0.45 + k / 3.0, 1.0);
            char z[2] = "Z";
            cairo_save(cr);
            text_path(cr, "Anton", 0, headx - d * 10 + u * 34 * -d + sin(u * 7) * 5, heady + 10 - u * 60, 16 + u * 14, 0, 0, z, NULL);
            cairo_restore(cr);
            cairo_set_line_width(cr, 4);
            cairo_set_source_rgba(cr, INK.r, INK.g, INK.b, 1 - u);
            cairo_stroke_preserve(cr);
            cairo_set_source_rgba(cr, 0.85, 0.92, 1, 1 - u);
            cairo_fill(cr);
        }
    }
    if (f->dizzy > 0.5) {
        for (int k = 0; k < 3; k++) {
            double a = t * 4 + k * 2 * M_PI / 3;
            star_path(cr, headx + cos(a) * 34, heady + 18 + sin(a) * 9, 9);
            cairo_set_source_rgb(cr, INK.r, INK.g, INK.b);
            cairo_set_line_width(cr, 3);
            cairo_stroke_preserve(cr);
            set_rgb(cr, GOLD);
            cairo_fill(cr);
        }
    }
}

static void draw_beam(cairo_t *cr, double x0, double x1, double y, double th, rgb col, double t)
{
    if (x1 < x0) {
        double tmp = x0;
        x0 = x1;
        x1 = tmp;
    }
    th *= 1 + 0.10 * sin(t * 37) + 0.05 * sin(t * 23);
    cairo_pattern_t *g = cairo_pattern_create_linear(0, y - th, 0, y + th);
    cairo_pattern_add_color_stop_rgba(g, 0.00, col.r, col.g, col.b, 0);
    cairo_pattern_add_color_stop_rgba(g, 0.22, col.r, col.g, col.b, 0.55);
    cairo_pattern_add_color_stop_rgba(g, 0.38, col.r, col.g, col.b, 1);
    cairo_pattern_add_color_stop_rgba(g, 0.46, 1, 1, 1, 1);
    cairo_pattern_add_color_stop_rgba(g, 0.54, 1, 1, 1, 1);
    cairo_pattern_add_color_stop_rgba(g, 0.62, col.r, col.g, col.b, 1);
    cairo_pattern_add_color_stop_rgba(g, 0.78, col.r, col.g, col.b, 0.55);
    cairo_pattern_add_color_stop_rgba(g, 1.00, col.r, col.g, col.b, 0);
    cairo_rectangle(cr, x0, y - th, x1 - x0, 2 * th);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    /* energy ripples travelling along the beam */
    cairo_set_line_width(cr, 3);
    cairo_set_source_rgba(cr, 1, 1, 1, 0.55);
    for (double x = x0 + fmod(t * 400, 26); x < x1; x += 26) {
        cairo_move_to(cr, x, y - th * 0.45);
        cairo_line_to(cr, x + 8, y + th * 0.45);
    }
    cairo_stroke(cr);
}

static void sprite(cairo_t *cr, cairo_surface_t *s, double x, double y, double size, double alpha)
{
    double w = cairo_image_surface_get_width(s), k = size / w;
    cairo_save(cr);
    cairo_translate(cr, x - size / 2, y - size / 2);
    cairo_scale(cr, k, k);
    cairo_set_source_surface(cr, s, 0, 0);
    cairo_paint_with_alpha(cr, alpha);
    cairo_restore(cr);
}

static void draw_effects(cairo_t *cr, const stats *s, double t)
{
    int b0 = F[0].beam > 0.5, b1 = F[1].beam > 0.5;

    /* beams from the front glove to the clash point or to the other fighter */
    for (int i = 0; i < 2; i++) {
        fighter *f = &F[i];
        if (f->beam < 0.05)
            continue;
        int o = 1 - i;
        double d = i == 0 ? 1 : -1;
        double end = (b0 && b1) ? clash_x : fighter_x(o) - d * 44;
        double th = (12 + 22 * clamp01((s->tok_port[i] - BEAM_TOK) / 900)) * f->beam;
        double x0 = f->hand_x + d * 8;
        if ((end - x0) * d > 0)
            draw_beam(cr, x0, end, f->hand_y, th, *team(i), t + i);
        sprite(cr, fb_sprite[i], f->hand_x + d * 6, f->hand_y, (40 + th) * (1 + 0.1 * sin(t * 30)), f->beam);
        if (!(b0 && b1))
            sprite(cr, fb_sprite[i], end, f->hand_y, 70 + th, 0.9 * f->beam);
    }
    if (b0 && b1) {
        double y = (F[0].hand_y + F[1].hand_y) / 2, r = 34 + 8 * sin(t * 25) + fmin(26, s->tok_s / 90);
        cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
        for (int k = 0; k < 12; k++) {
            double a = t * 2.5 + k * M_PI / 6, l = r * (1.5 + 0.5 * sin(t * 13 + k * 1.9));
            cairo_move_to(cr, clash_x + cos(a) * r * 0.4, y + sin(a) * r * 0.4);
            cairo_line_to(cr, clash_x + cos(a) * l, y + sin(a) * l);
            set_rgb(cr, k & 1 ? BLUE : ORANGE);
            cairo_set_line_width(cr, 4);
            cairo_stroke(cr);
        }
        sprite(cr, clash_sprite, clash_x, y, r * 2.6, 1);
    }

    for (int j = 0; j < MAX_FB; j++) {
        const fireball *fb = &fbs[j];
        if (!fb->alive)
            continue;
        double d = fb->side == 0 ? 1 : -1, y = FB_Y + sin(fb->age * 25) * 2;
        for (int k = 2; k >= 0; k--)
            sprite(cr, fb_sprite[fb->side], fb->x - d * k * 15, y, 54 - k * 12, k ? 0.35 : 1);
    }

    for (int i = 0; i < MAX_PART; i++) {
        const particle *p = &parts[i];
        if (!p->alive || p->type == P_ROCK)
            continue;
        double u = p->age / p->life;
        if (p->type == P_SPARK) {
            cairo_move_to(cr, p->x, p->y);
            cairo_line_to(cr, p->x - p->vx * 0.035, p->y - p->vy * 0.035);
            cairo_set_source_rgba(cr, p->col.r, p->col.g, p->col.b, 1 - u);
            cairo_set_line_width(cr, 3.5);
            cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
            cairo_stroke(cr);
        } else {
            cairo_new_path(cr);
            cairo_arc(cr, p->x, p->y, p->size * (1 + 1.6 * u), 0, 2 * M_PI);
            cairo_set_source_rgba(cr, 0.55, 0.55, 0.58, 0.55 * (1 - u));
            cairo_fill(cr);
        }
    }

    for (int i = 0; i < MAX_WORDS; i++) {
        const hitword *w = &words[i];
        if (!w->alive)
            continue;
        double sc = 1 + 0.7 * exp(-w->age * 16), a = w->age > 0.5 ? 1 - (w->age - 0.5) / 0.2 : 1;
        cairo_save(cr);
        cairo_translate(cr, w->x, w->y - w->age * 20);
        cairo_rotate(cr, w->rot);
        cairo_scale(cr, sc, sc);
        text_path(cr, "Anton", 0, 0, 0, 30, 0, 0, w->word, NULL);
        cairo_restore(cr);
        cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
        cairo_set_line_width(cr, 6);
        cairo_set_source_rgba(cr, 0.55, 0.02, 0.05, a);
        cairo_stroke_preserve(cr);
        cairo_set_source_rgba(cr, 1, 0.92, 0.2, a);
        cairo_fill(cr);
    }

    /* lightning past ULTRA_TOK */
    if (t - bolt_t < 0.13) {
        unsigned seed = bolt_seed;
        double x = 120 + (rand_r(&seed) % 240), y = 120;
        double tx = F[rand_r(&seed) & 1].hand_x;
        cairo_new_path(cr);
        cairo_move_to(cr, x, y);
        while (y < FB_Y) {
            y += 18 + rand_r(&seed) % 16;
            x += (tx - x) * 0.25 + (int)(rand_r(&seed) % 40) - 20;
            cairo_line_to(cr, x, y);
        }
        cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
        cairo_set_line_width(cr, 10);
        cairo_set_source_rgba(cr, 0.6, 0.7, 1, 0.35);
        cairo_stroke_preserve(cr);
        cairo_set_line_width(cr, 3);
        cairo_set_source_rgb(cr, 1, 1, 1);
        cairo_stroke(cr);
    }
}

static void draw_rocks(cairo_t *cr)
{
    for (int i = 0; i < MAX_PART; i++) {
        const particle *p = &parts[i];
        if (!p->alive || p->type != P_ROCK)
            continue;
        double u = p->age / p->life, a = fmin(1, fmin(u * 5, (1 - u) * 4));
        cairo_save(cr);
        cairo_translate(cr, p->x, p->y);
        cairo_rotate(cr, p->rot);
        cairo_new_path(cr);
        cairo_move_to(cr, -p->size, -p->size * 0.4);
        cairo_line_to(cr, -p->size * 0.2, -p->size);
        cairo_line_to(cr, p->size, -p->size * 0.3);
        cairo_line_to(cr, p->size * 0.6, p->size * 0.8);
        cairo_line_to(cr, -p->size * 0.7, p->size * 0.7);
        cairo_close_path(cr);
        cairo_restore(cr);
        cairo_set_source_rgba(cr, 0.30, 0.28, 0.38, a);
        cairo_fill_preserve(cr);
        cairo_set_source_rgba(cr, 0.05, 0.03, 0.10, a);
        cairo_set_line_width(cr, 2);
        cairo_stroke(cr);
    }
}

typedef struct { double tok, tok0, tok1; } shown_t;


static void update_hud(const stats *s, const shown_t *sh, double t)
{
    static double tok, tok0, tok1, watts, next;
    static int combo[2];
    char key[256], txt[64];
    int blink = (int)(t * 2) & 1;

    if (t >= next || t < next - 1) {
        next = t + 0.25;
        tok = shown_rate(sh->tok);
        tok0 = sh->tok0;
        tok1 = sh->tok1;
        watts = s->power[0] + s->power[1];
        for (int i = 0; i < 2; i++)
            combo[i] = t - F[i].combo_t < 2.0 && F[i].combo >= 2 ? F[i].combo : 0;
    }
    int over = watts > OVER_W, idle_text = rstate == R_IDLE && !ann_on;
    snprintf(key, sizeof(key), "%.0f|%.0f|%.0f|%.0f|%d|%d|%d|%d|%d|%d|%d|%d", tok, tok0, tok1, watts, s->temp[0],
             s->temp[1], combo[0], combo[1], s->running, over ? blink : 2, idle_text ? blink : 2, rstate == R_IDLE);
    if (!strcmp(key, hud_key))
        return;
    snprintf(hud_key, sizeof(hud_key), "%s", key);

    cairo_t *cr = cairo_create(hud_cache);
    cairo_set_operator(cr, CAIRO_OPERATOR_CLEAR);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);

    /* names */
    arcade_text(cr, "Anton", 0, 129, 72, 24, 0, 0, BLUE, NAME[0]);
    arcade_text(cr, "Anton", 0, 351, 72, 24, 0, 0, ORANGE, NAME[1]);

    /* power bars: per-server tok/s, anchored at the timer */
    for (int i = 0; i < 2; i++) {
        double v = i ? tok1 : tok0, inner = i ? 284 : 196, outer = i ? 418 : 62;
        double x0 = fmin(inner, outer), w = fabs(inner - outer), y = 88, h = 26;
        double fill = clamp01(v / BAR_MAX) * w;
        rgb tc = *team(i);
        cairo_rectangle(cr, x0 - 3, y - 3, w + 6, h + 6);
        set_rgb(cr, INK);
        cairo_fill(cr);
        cairo_rectangle(cr, x0, y, w, h);
        cairo_set_source_rgb(cr, 0.22, 0.05, 0.08);
        cairo_fill(cr);
        if (fill > 0.5) {
            cairo_pattern_t *g = cairo_pattern_create_linear(0, y, 0, y + h);
            rgb top = v >= BAR_MAX ? GOLD : lerp(tc, (rgb){ 1, 1, 1 }, 0.35);
            rgb bot = v >= BAR_MAX ? (rgb){ 0.9, 0.45, 0.05 } : lerp(tc, INK, 0.25);
            cairo_pattern_add_color_stop_rgb(g, 0, top.r, top.g, top.b);
            cairo_pattern_add_color_stop_rgb(g, 1, bot.r, bot.g, bot.b);
            cairo_rectangle(cr, i ? inner : inner - fill, y, fill, h);
            cairo_set_source(cr, g);
            cairo_fill(cr);
            cairo_pattern_destroy(g);
        }
        cairo_rectangle(cr, x0 - 1.5, y - 1.5, w + 3, h + 3);
        cairo_set_source_rgb(cr, 0.95, 0.85, 0.55);
        cairo_set_line_width(cr, 2);
        cairo_stroke(cr);
        snprintf(txt, sizeof(txt), "%.0f", shown_gpu_rate(v));
        arcade_text(cr, "Anton", 0, i ? inner + 8 : inner - 8, y + h / 2, 24, 0, i ? -1 : 1, (rgb){ 1, 1, 1 }, txt);
        if (combo[i]) {
            snprintf(txt, sizeof(txt), "%d HITS", combo[i]);
            arcade_text(cr, "Anton", 0, i ? 412 : 68, 136, 26, 0, i ? 1 : -1, GOLD, txt);
        }
    }

    /* the round timer is total tok/s */
    rounded_rect(cr, 196, 74, 88, 54, 8);
    cairo_set_source_rgba(cr, 0.06, 0.02, 0.10, 0.92);
    cairo_fill_preserve(cr);
    cairo_set_line_width(cr, 3);
    cairo_set_source_rgb(cr, 0.95, 0.85, 0.55);
    cairo_stroke(cr);
    snprintf(txt, sizeof(txt), "%.0f", tok);
    arcade_text(cr, "Anton", 0, 240, 101, 40, 76, 0, GOLD, txt);
    arcade_text(cr, "DejaVu Sans", 1, 240, 142, 15, 0, 0, (rgb){ 0.9, 0.9, 1 }, rate_unit_uc());

    /* bottom: temps under each fighter, total watts, running requests as credits */
    for (int i = 0; i < 2; i++) {
        snprintf(txt, sizeof(txt), "%d\xC2\xB0", s->temp[i]);
        rgb c = s->temp[i] >= TOASTY_C ? (rgb){ 1, 0.35, 0.25 } : *team(i);
        arcade_text(cr, "Anton", 0, i ? 334 : 146, 426, 28, 0, 0, c, txt);
    }
    snprintf(txt, sizeof(txt), "%.0f W", watts);
    arcade_text(cr, "Anton", 0, 240, 426, 28, 0, 0, over && blink ? (rgb){ 1, 0.25, 0.2 } : (rgb){ 1, 1, 1 }, txt);
    snprintf(txt, sizeof(txt), "CREDITS %d", s->running);
    arcade_text(cr, "DejaVu Sans", 1, 240, 455, 15, 0, 0, (rgb){ 0.85, 0.85, 0.95 }, txt);

    if (idle_text && blink)
        arcade_text(cr, "Anton", 0, 240, 200, 44, 0, 0, GOLD, "INSERT PROMPT");
    if (idle_text)
        arcade_text(cr, "DejaVu Sans", 1, 240, 240, 20, 0, 0, (rgb){ 1, 1, 1 }, "TO CONTINUE");
    cairo_destroy(cr);
}

static void draw_announcer(cairo_t *cr, double t)
{
    if (!ann_on)
        return;
    double age = t - ann_start, left = ann_cur.dur - age;
    double sc = 1 + 0.3 * exp(-age * 14), a = clamp01(left / 0.18);
    double sz, y = 196;
    double max_w = chord_width(y, 40) - 24;
    cairo_save(cr);
    cairo_translate(cr, SIZE / 2.0, y);
    cairo_scale(cr, sc, sc);
    text_path(cr, "Anton", 0, 0, 0, 50, max_w, 0, ann_cur.main, &sz);
    cairo_restore(cr);
    cairo_pattern_t *g = cairo_pattern_create_linear(0, y - sz * 0.4 * sc, 0, y + sz * 0.4 * sc);
    cairo_pattern_add_color_stop_rgba(g, 0, 1, 0.97, 0.55, a);
    cairo_pattern_add_color_stop_rgba(g, 0.5, 1, 0.75, 0.10, a);
    cairo_pattern_add_color_stop_rgba(g, 1, 0.95, 0.20, 0.05, a);
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
    cairo_set_line_width(cr, sz * sc * 0.18);
    cairo_set_source_rgba(cr, 0.12, 0.0, 0.03, a);
    cairo_stroke_preserve(cr);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    if (ann_cur.sub[0] && a > 0.05) {
        text_path(cr, "DejaVu Sans", 1, SIZE / 2.0, y + 40, 22, chord_width(y + 40, 22) - 30, 0, ann_cur.sub, &sz);
        cairo_set_line_width(cr, 5);
        cairo_set_source_rgba(cr, INK.r, INK.g, INK.b, a);
        cairo_stroke_preserve(cr);
        cairo_set_source_rgba(cr, 1, 1, 1, a);
        cairo_fill(cr);
    }
}

static void draw_toasty(cairo_t *cr, double t)
{
    double age = t - toasty_start;
    if (age < 0 || age > 1.6)
        return;
    double u = age < 0.2 ? 1 - age / 0.2 : age > 1.35 ? (age - 1.35) / 0.25 : 0;
    u = u * u;
    double w = cairo_image_surface_get_width(toasty_img);
    cairo_set_source_surface(cr, toasty_img, 312 + u * 110, 300 + u * 110);
    cairo_paint(cr);
    if (u < 0.3) {
        cairo_save(cr);
        cairo_translate(cr, 312 - 4, 290);
        cairo_rotate(cr, -0.12);
        arcade_text(cr, "Anton", 0, 0, 0, 36, 0, 0, (rgb){ 1, 0.55, 0.15 }, "TOASTY!");
        cairo_restore(cr);
    }
    (void)w;
}

static void render(cairo_t *cr, const stats *s, const shown_t *sh, double t)
{
    cairo_set_source_surface(cr, stage_img, 0, 0);
    cairo_paint(cr);

    draw_rocks(cr);
    cairo_save(cr);
    cairo_translate(cr, shake_x, shake_y);
    for (int i = 0; i < 2; i++)
        draw_fighter(cr, i, t);
    draw_effects(cr, s, t);
    cairo_restore(cr);
    if (t - bolt_t < 0.06) {
        cairo_set_source_rgba(cr, 0.8, 0.85, 1, 0.18);
        cairo_paint(cr);
    }
    draw_toasty(cr, t);

    update_hud(s, sh, t);
    cairo_set_source_surface(cr, hud_cache, 0, 0);
    cairo_paint(cr);
    draw_announcer(cr, t);
}

/* ---------------------------------------------------------------- showcase */

/* A scripted 44 second fight that hits every stage, for filming and the GIF */
typedef struct { double t, tok0, tok1; } keyframe;

static const keyframe script[] = {
    {  0.0,    0,    0 },                   /* both asleep: INSERT PROMPT */
    {  4.0,    0,    0 },
    {  4.3,   90,    0 },                   /* ROUND n: ZOTAC slaps a sleeping TUF */
    {  9.0,  170,    0 },
    {  9.5,  180,  150 },                   /* TUF wakes up: fireballs clash */
    { 14.5,  260,  220 },
    { 16.5,  430,  380 },                   /* beam struggle */
    { 20.0,  470,  720 },                   /* TUF pushes the clash toward ZOTAC, goes super */
    { 23.5,  940,  700 },                   /* ZOTAC pushes back */
    { 26.5, 1250, 1150 },                   /* ultra: shaking, rocks, lightning, 900 W, toasty */
    { 32.0, 1250, 1150 },
    { 33.5,    0,    0 },                   /* K.O., winner, dizzy loser */
    { 44.0,    0,    0 },
};
#define SCRIPT_LEN      (sizeof(script) / sizeof(script[0]))
#define SCRIPT_PERIOD   44.0

static void showcase_poll(stats *s, double t)
{
    double lt = fmod(t, SCRIPT_PERIOD), tok[2] = { 0, 0 };
    for (size_t i = 0; i + 1 < SCRIPT_LEN; i++) {
        const keyframe *a = &script[i], *b = &script[i + 1];
        if (lt >= a->t && lt < b->t) {
            double u = (lt - a->t) / (b->t - a->t);
            u = u * u * (3 - 2 * u);
            tok[0] = a->tok0 + (b->tok0 - a->tok0) * u;
            tok[1] = a->tok1 + (b->tok1 - a->tok1) * u;
            break;
        }
    }
    s->tok_s = 0;
    s->running = 0;
    for (int i = 0; i < 2; i++) {
        double wob = 1 + 0.04 * sin(t * (2.3 + i));
        s->tok_port[i] = tok[i] < 1 ? 0 : tok[i] * wob;
        s->tok_s += s->tok_port[i];
        s->running_port[i] = tok[i] >= 1 ? 1 + (int)(tok[i] / 250) : 0;
        s->running += s->running_port[i];
        s->load[i] = clamp01(tok[i] / 700);
        s->power[i] = 28 + i * 5 + fmin(545, tok[i] * 0.47);
        s->temp[i] = 36 + i * 5 + (int)fmin(46, tok[i] / 28);
    }
}

/* ---------------------------------------------------------------- main */

/* --demo: the shared demo_poll, scaled up (on a copy) until it reaches every stage */
static void demo_scaled(stats *s, double t)
{
    static stats base;
    demo_poll(&base, t);
    *s = base;
    s->tok_port[0] *= 9;
    s->tok_port[1] *= 9;
    s->tok_s = s->tok_port[0] + s->tok_port[1];
    if (s->tok_s < 400) {
        s->tok_s = s->tok_port[0] = s->tok_port[1] = 0;
        s->running = 0;
    }
    s->running = 0;
    for (int i = 0; i < 2; i++) {
        if (s->tok_port[i] < 150)
            s->tok_s -= s->tok_port[i], s->tok_port[i] = 0;
        s->running_port[i] = s->tok_port[i] > 0 ? 1 + (int)(s->tok_port[i] / 300) : 0;
        s->running += s->running_port[i];
        s->power[i] *= 1.25;
    }
}

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

    const char *source = getenv("LLM_REACTOR_SOURCE");
    gpu_source = source && !strcasecmp(source, "gpu");
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gpu-load"))
            gpu_source = 1;
        else if (!strcmp(argv[i], "--demo"))
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
    load_assets();
    reset_scene();

    surf = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cr = cairo_create(surf);
    tj = tj3Init(TJINIT_COMPRESS);
    tj3Set(tj, TJPARAM_QUALITY, JPEG_QUALITY);
    tj3Set(tj, TJPARAM_SUBSAMP, TJSAMP_420);

    if (bench) {
        struct { double tok0, tok1, w0, w1; int t0, t1; double secs; const char *png; } scenes[] = {
            { 1180, 1020, 560, 548, 76, 83, 14.0, "kombat_preview.png" },  /* ultra beam struggle */
            {  210,  150, 260, 190, 55, 61, 12.6, "kombat_mid.png" },      /* fireball fight */
            {  460,    0, 330,  30, 63, 40, 9.25, "kombat_solo.png" },      /* ZOTAC beams a sleeping TUF */
            {    0,    0,  25,  30, 36, 41, 12.0, "kombat_idle.png" },     /* INSERT PROMPT */
        };
        for (int k = 0; k < 4; k++) {
            reset_scene();
            memset(&s, 0, sizeof(s));
            srand(7);
            s.tok_port[0] = scenes[k].tok0;
            s.tok_port[1] = scenes[k].tok1;
            s.tok_s = s.tok_port[0] + s.tok_port[1];
            for (int i = 0; i < 2; i++) {
                s.running_port[i] = s.tok_port[i] > 0 ? 2 : 0;
                s.load[i] = clamp01(s.tok_port[i] / 700);
            }
            s.running = s.running_port[0] + s.running_port[1];
            s.power[0] = scenes[k].w0;
            s.power[1] = scenes[k].w1;
            s.temp[0] = scenes[k].t0;
            s.temp[1] = scenes[k].t1;
            sh.tok = s.tok_s;
            sh.tok0 = s.tok_port[0];
            sh.tok1 = s.tok_port[1];
            int n = (int)(scenes[k].secs * FPS_BUSY);
            size_t len = 0;
            double b0 = 0, worst = 0;
            for (int i = 0; i < n; i++) {
                double f0 = now_s();
                if (i == n - 60)
                    b0 = f0;
                simulate(&s, 1.0 / FPS_BUSY, i / (double)FPS_BUSY);
                render(cr, &s, &sh, i / (double)FPS_BUSY);
                cairo_surface_flush(surf);
                tj3Compress8(tj, cairo_image_surface_get_data(surf), SIZE, cairo_image_surface_get_stride(surf),
                             SIZE, TJPF_BGRX, &jpeg, &len);
                if (i >= n - 60)
                    worst = fmax(worst, now_s() - f0);
            }
            printf("%s: %.2f ms/frame (worst %.2f), jpeg %zu bytes\n", scenes[k].png, (now_s() - b0) * 1000 / 60,
                   worst * 1000, len);
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
                demo_scaled(&s, t - t0);
            } else {
                gpus_poll(&s);
                if (gpu_source)
                    gpu_rate_poll(&s);
                else
                    vllm_poll(&s, t);
            }
        }

        k = showcase ? 1 : fmin(1, dt * 4);
        sh.tok += (s.tok_s - sh.tok) * k;
        sh.tok0 += (s.tok_port[0] - sh.tok0) * k;
        sh.tok1 += (s.tok_port[1] - sh.tok1) * k;

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

        /* Slow down while everyone is asleep */
        int n_parts = 0;
        for (int i = 0; i < MAX_PART; i++)
            n_parts += parts[i].alive;
        int idle = !showcase && rstate == R_IDLE && !ann_on && n_parts == 0 && F[0].sleep > 0.95 && F[1].sleep > 0.95;
        double spare = 1.0 / (showcase ? FPS_SHOWCASE : idle ? FPS_IDLE : FPS_BUSY) - (now_s() - t);
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
