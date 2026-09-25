/*
 * thirst: two GPUs chase kids to steal their water, on the iCUE LINK AIO pump LCD.
 * A satire of AI datacenters' water use.
 *
 * Kids in a parched town carry cups of water. Each GPU (blue for GPU 0, orange for
 * GPU 1) hunts the nearest kid who still has some, at a speed set by its vLLM
 * server's tokens/sec: at low load the kids outrun them, at high load they don't.
 * Caught, the GPU slurps the cup dry and the kid trudges to the well to refill.
 * The town water tower drains as tokens flow; idle, the GPUs nap by the datacenter.
 * The counter is a datacenter-equivalent estimate (see ML_PER_TOKEN), not a
 * measurement: this machine's own loop is closed and uses no water.
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
        /*
         * A failed or cut-short fetch (no counter in it) keeps the last good sample
         * as the baseline, so the next good one gives the rate over the whole gap
         * instead of a zero followed by a spike.
         */
        if (http_metrics(vllm_ports[i], buf, 1 << 20) <= 0 || !strstr(buf, "\nvllm:generation_tokens_total")) {
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
#define N_KIDS          4
#define GROUND_Y        262.0       /* where the dry ground starts */
#define ARENA_X0        78.0
#define ARENA_X1        402.0
#define ARENA_Y0        318.0
#define ARENA_Y1        404.0
#define WELL_X          240.0
#define WELL_Y          300.0
#define DOOR_X          120.0       /* where GPUs nap, in front of the datacenter */
#define DOOR_Y          330.0
/*
 * Water per generated token, as a datacenter-equivalent estimate: Li et al. 2023,
 * "Making AI Less Thirsty", put GPT-3 at roughly 500 mL per 10 to 50 medium-length
 * responses. Using 30 responses of ~300 tokens: 500 / 9000 = 0.056 mL per token.
 */
#define ML_PER_TOKEN    (500.0 / (30 * 300))
#define TANK_L          3.0         /* litres the water tower shows as full */
#define KID_SCALE       2.0
#define GPU_SCALE       1.7
#define BG_SHIFT        (-58.0)     /* datacenter, tower and sun sit higher than the original layout */

enum { KID_WATER, KID_DRAINED, KID_REFILL };
enum { GPU_HUNT, GPU_DRINK, GPU_NAP };

/*
 * Everyone moves with a smoothed velocity (vx, vy) that eases toward where they
 * want to go, and faces the way that velocity points. Facing only flips after the
 * velocity has clearly pointed the other way for a moment (turn), and never
 * sooner than FACE_HOLD seconds after the last flip (since).
 */
#define FACE_VX         12.0        /* px/s of backwards motion that counts as turning */
#define FACE_TURN       0.25        /* s of turning before the sprite flips */
#define FACE_HOLD       0.7         /* s between flips at the least */

typedef struct {
    double x, y, vx, vy, water, step, face, turn, since;
    double dx, dy;                  /* wander direction */
    double hx, hy;                  /* where they head after refilling */
    double safe;                    /* head start after refilling: GPUs ignore them */
    double timer;
    int    state, shirt, skin, homing;
} kid;

typedef struct {
    double x, y, vx, vy, face, turn, since, fan, step, timer, speed;
    int    state, target;
} gpu;

static kid    kids[N_KIDS];
static gpu    gpus[2];
static double liters, tank = 1.0;
static cairo_surface_t *bg_cache, *hud_cache;
static char hud_key[128];

static double frand(void) { return rand() / (double)RAND_MAX; }

static const rgb SHIRTS[5] = { { 0.95, 0.35, 0.35 }, { 0.30, 0.70, 0.40 }, { 0.95, 0.80, 0.25 }, { 0.55, 0.45, 0.90 }, { 0.30, 0.65, 0.90 } };
static const rgb SKINS[3] = { { 0.96, 0.80, 0.66 }, { 0.78, 0.56, 0.40 }, { 0.50, 0.34, 0.24 } };
static const rgb GPU_COL[2] = { { 0.20, 0.50, 0.95 }, { 0.98, 0.52, 0.12 } };
static const rgb WATER = { 0.30, 0.65, 1.0 };

/* ---------------------------------------------------------------- static background */

static void build_background(void)
{
    const double c = SIZE / 2.0;
    srand(21);
    bg_cache = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cairo_t *cr = cairo_create(bg_cache);

    /* Hot, hazy sky */
    cairo_pattern_t *g = cairo_pattern_create_linear(0, 0, 0, GROUND_Y);
    cairo_pattern_add_color_stop_rgb(g, 0, 0.98, 0.80, 0.52);
    cairo_pattern_add_color_stop_rgb(g, 1, 1.00, 0.92, 0.72);
    cairo_rectangle(cr, 0, 0, SIZE, GROUND_Y);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    cairo_save(cr);
    cairo_translate(cr, 0, BG_SHIFT);
    /* Blazing sun */
    g = cairo_pattern_create_radial(372, 118, 20, 372, 118, 110);
    cairo_pattern_add_color_stop_rgba(g, 0, 1, 1, 0.85, 0.9);
    cairo_pattern_add_color_stop_rgba(g, 1, 1, 0.85, 0.5, 0);
    cairo_arc(cr, 372, 118, 110, 0, 2 * M_PI);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    cairo_arc(cr, 372, 118, 26, 0, 2 * M_PI);
    cairo_set_source_rgb(cr, 1, 0.97, 0.80);
    cairo_fill(cr);

    cairo_restore(cr);

    /* Dry, cracked ground */
    cairo_rectangle(cr, 0, GROUND_Y, SIZE, SIZE - GROUND_Y);
    g = cairo_pattern_create_linear(0, GROUND_Y, 0, SIZE);
    cairo_pattern_add_color_stop_rgb(g, 0, 0.86, 0.68, 0.44);
    cairo_pattern_add_color_stop_rgb(g, 1, 0.70, 0.50, 0.30);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    cairo_set_source_rgba(cr, 0.45, 0.30, 0.16, 0.55);
    cairo_set_line_width(cr, 1.3);
    for (int i = 0; i < 38; i++) {
        double x = frand() * SIZE, y = GROUND_Y + 8 + frand() * (SIZE - GROUND_Y);
        cairo_move_to(cr, x, y);
        for (int k = 0; k < 4; k++) {
            x += (frand() - 0.5) * 30;
            y += (frand() - 0.3) * 12;
            cairo_line_to(cr, x, y);
        }
    }
    cairo_stroke(cr);
    /* a dead shrub or two */
    cairo_set_source_rgb(cr, 0.45, 0.32, 0.18);
    cairo_set_line_width(cr, 1.6);
    for (int b = 0; b < 3; b++) {
        double bx = 60 + b * 180 + frand() * 20, by = GROUND_Y + 10 + b * 4;
        for (int k = 0; k < 6; k++) {
            double a = -M_PI / 2 + (frand() - 0.5) * 1.8;
            cairo_move_to(cr, bx, by);
            cairo_line_to(cr, bx + cos(a) * 12, by + sin(a) * 12);
        }
    }
    cairo_stroke(cr);

    cairo_save(cr);
    cairo_translate(cr, 0, BG_SHIFT);
    /* The datacenter: grey box, vents, a big "AI" logo, cooling units on the roof */
    cairo_rectangle(cr, 34, 214, 150, 110);
    cairo_set_source_rgb(cr, 0.55, 0.57, 0.62);
    cairo_fill(cr);
    cairo_rectangle(cr, 34, 214, 150, 12);
    cairo_set_source_rgb(cr, 0.42, 0.44, 0.50);
    cairo_fill(cr);
    for (int u = 0; u < 3; u++) {
        cairo_rectangle(cr, 46 + u * 44, 196, 34, 18);
        cairo_set_source_rgb(cr, 0.48, 0.50, 0.56);
        cairo_fill(cr);
        cairo_arc(cr, 63 + u * 44, 205, 6, 0, 2 * M_PI);
        cairo_set_source_rgb(cr, 0.30, 0.32, 0.36);
        cairo_fill(cr);
    }
    cairo_set_source_rgb(cr, 0.40, 0.42, 0.48);
    for (int v = 0; v < 5; v++) {
        cairo_rectangle(cr, 46, 238 + v * 10, 60, 4);
    }
    cairo_fill(cr);
    cairo_rectangle(cr, 108, 294, 26, 30);                 /* door */
    cairo_set_source_rgb(cr, 0.30, 0.31, 0.35);
    cairo_fill(cr);
    cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, 20);
    cairo_move_to(cr, 140, 285);
    cairo_set_source_rgb(cr, 0.25, 0.85, 1.0);
    cairo_show_text(cr, "AI");

    /* The town water tower (tank drawn per frame so the level can drop) */
    cairo_set_source_rgb(cr, 0.42, 0.30, 0.22);
    cairo_set_line_width(cr, 4);
    cairo_move_to(cr, 370, 266); cairo_line_to(cr, 358, 324);
    cairo_move_to(cr, 414, 266); cairo_line_to(cr, 426, 324);
    cairo_move_to(cr, 362, 296); cairo_line_to(cr, 422, 296);
    cairo_stroke(cr);

    cairo_restore(cr);

    /* The village well the kids refill at */
    cairo_save(cr);
    cairo_translate(cr, WELL_X, WELL_Y);
    cairo_scale(cr, 1, 0.45);
    cairo_arc(cr, 0, 0, 20, 0, 2 * M_PI);
    cairo_restore(cr);
    cairo_set_source_rgb(cr, 0.55, 0.52, 0.50);
    cairo_fill(cr);
    cairo_rectangle(cr, WELL_X - 20, WELL_Y - 18, 40, 18);
    cairo_set_source_rgb(cr, 0.62, 0.58, 0.55);
    cairo_fill(cr);
    cairo_set_source_rgb(cr, 0.45, 0.30, 0.18);
    cairo_rectangle(cr, WELL_X - 20, WELL_Y - 44, 4, 28);
    cairo_rectangle(cr, WELL_X + 16, WELL_Y - 44, 4, 28);
    cairo_rectangle(cr, WELL_X - 24, WELL_Y - 48, 48, 6);
    cairo_fill(cr);

    /* soft round vignette */
    g = cairo_pattern_create_radial(c, c, 190, c, c, 250);
    cairo_pattern_add_color_stop_rgba(g, 0, 0.3, 0.15, 0.05, 0);
    cairo_pattern_add_color_stop_rgba(g, 1, 0.3, 0.15, 0.05, 0.55);
    cairo_set_source(cr, g);
    cairo_paint(cr);
    cairo_pattern_destroy(g);
    cairo_destroy(cr);

    hud_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, SIZE);
    srand((unsigned)time(NULL));
}

/* ---------------------------------------------------------------- simulation */

static void init_actors(void)
{
    for (int i = 0; i < N_KIDS; i++) {
        kid *k = &kids[i];
        memset(k, 0, sizeof(*k));
        k->x = ARENA_X0 + 30 + frand() * (ARENA_X1 - ARENA_X0 - 60);
        k->y = ARENA_Y0 + frand() * (ARENA_Y1 - ARENA_Y0);
        k->water = 1;
        k->state = KID_WATER;
        k->shirt = i % 5;
        k->skin = rand() % 3;
        k->face = frand() < 0.5 ? -1 : 1;
        k->since = FACE_HOLD;
    }
    for (int g = 0; g < 2; g++) {
        memset(&gpus[g], 0, sizeof(gpus[g]));
        gpus[g].x = DOOR_X + g * 150;
        gpus[g].y = DOOR_Y + g * 12;
        gpus[g].face = 1;
        gpus[g].since = FACE_HOLD;
        gpus[g].state = GPU_NAP;
        gpus[g].target = -1;
    }
}

/* Keep inside the arena, dropping any velocity that pushes into the edge. */
static void clamp_arena(double *x, double *y, double *vx, double *vy)
{
    if (*x < ARENA_X0 || *x > ARENA_X1) {
        *x = fmax(ARENA_X0, fmin(ARENA_X1, *x));
        if (vx)
            *vx = 0;
    }
    if (*y < ARENA_Y0 || *y > ARENA_Y1) {
        *y = fmax(ARENA_Y0, fmin(ARENA_Y1, *y));
        if (vy)
            *vy = 0;
    }
}

/* Ease velocity toward the wanted one (a fifth of a second time constant) and move. */
static void steer(double *x, double *y, double *vx, double *vy, double wvx, double wvy, double dt)
{
    double k = fmin(1, dt * 5);
    *vx += (wvx - *vx) * k;
    *vy += (wvy - *vy) * k;
    *x += *vx * dt;
    *y += *vy * dt;
    clamp_arena(x, y, vx, vy);
}

/* Flip to face the way we move, only once it is clear we turned around. */
static void update_face(double *face, double *turn, double *since, double vx, double dt)
{
    *since += dt;
    if (vx * *face < -FACE_VX)
        *turn += dt;
    else
        *turn = 0;
    if (*turn >= FACE_TURN && *since >= FACE_HOLD) {
        *face = -*face;
        *turn = 0;
        *since = 0;
    }
}

static int kid_huntable(int i, int gi)
{
    return kids[i].state == KID_WATER && kids[i].safe <= 0 &&
           !(gpus[!gi].state == GPU_DRINK && gpus[!gi].target == i);
}

/* How scary a spot is for a kid: closeness to each hunting GPU within 150 px. */
static double kid_threat(double x, double y)
{
    double sum = 0;
    for (int gi = 0; gi < 2; gi++)
        if (gpus[gi].state == GPU_HUNT)
            sum += fmax(0, 150 - hypot(x - gpus[gi].x, y - gpus[gi].y));
    return sum;
}

static void simulate(const stats *s, double dt, double t)
{
    int idle = s->tok_s < 1 && s->running == 0;

    /* Datacenter-equivalent water for the tokens generated */
    double used = s->tok_s * dt * ML_PER_TOKEN / 1000.0;
    liters += used;
    tank = fmax(0.04, tank - used / TANK_L);
    if (idle)
        tank = fmin(1, tank + dt * 0.01);

    /* GPUs: hunt kids who still have water, drink, nap when their server is idle */
    for (int gi = 0; gi < 2; gi++) {
        gpu *g = &gpus[gi];
        double tokp = s->tok_port[gi], wvx = 0, wvy = 0;
        g->fan += dt * (2 + tokp / 60);
        if (tokp < 1 && g->state != GPU_DRINK) {
            g->state = GPU_NAP;
            g->target = -1;
        } else if (g->state == GPU_NAP) {
            g->state = GPU_HUNT;
        }
        g->speed = 32 + fmin(tokp, 1400) * 0.14;

        if (g->state == GPU_DRINK) {
            kid *k = &kids[g->target];
            g->vx = g->vy = 0;
            k->water -= dt / 1.2;
            if (k->water <= 0) {
                k->water = 0;
                k->state = KID_DRAINED;
                g->state = GPU_HUNT;
                g->target = -1;
                g->timer = 0.9;                 /* a satisfied pause */
            }
            g->since += dt;
            continue;
        }
        if (g->state == GPU_NAP) {
            double dx = DOOR_X + gi * 150 - g->x, dy = DOOR_Y + gi * 12 - g->y, d = hypot(dx, dy);
            if (d > 3) {
                wvx = dx / d * 40 * fmin(1, d / 20);
                wvy = dy / d * 40 * fmin(1, d / 20);
            }
        } else if (g->timer > 0) {
            g->timer -= dt;
        } else {
            /*
             * Chase the nearest kid with water that the other GPU isn't drinking from,
             * but stick with the current one unless another is much closer.
             */
            int best = -1;
            double bd = 1e9, cd = 1e9;
            for (int i = 0; i < N_KIDS; i++) {
                if (!kid_huntable(i, gi))
                    continue;
                double d = hypot(kids[i].x - g->x, kids[i].y - g->y);
                if (d < bd) {
                    bd = d;
                    best = i;
                }
                if (i == g->target)
                    cd = d;
            }
            if (g->target < 0 || cd > 1e8 || bd < cd * 0.6)
                g->target = best;
            if (g->target >= 0) {
                kid *k = &kids[g->target];
                double dx = k->x - g->x, dy = k->y - g->y, d = hypot(dx, dy);
                /* caught: straw in the cup (a kid just behind waits for us to turn) */
                if (d < 50 && (fabs(dx) < 10 || dx * g->face > 0 || g->since >= FACE_HOLD)) {
                    g->state = GPU_DRINK;
                    if (fabs(dx) > 10 && dx * g->face < 0) {
                        g->face = -g->face;
                        g->since = g->turn = 0;
                    }
                    continue;
                }
                wvx = dx / d * g->speed;
                wvy = dy / d * g->speed;
            }
        }
        /* keep the two GPUs from standing inside each other */
        {
            gpu *o = &gpus[!gi];
            double dx = g->x - o->x, dy = g->y - o->y, d = hypot(dx, dy);
            if (d < 110 && d > 0.1 && g->state != GPU_NAP) {
                wvx += dx / d * (110 - d) * 1.5;
                wvy += dy / d * (110 - d) * 1.5;
            }
        }
        steer(&g->x, &g->y, &g->vx, &g->vy, wvx, wvy, dt);
        update_face(&g->face, &g->turn, &g->since, g->vx, dt);
        double v = hypot(g->vx, g->vy);
        if (v > 5)
            g->step += dt * (g->state == GPU_NAP ? 8 : 4 + v * 0.08);
        else
            g->step += dt * 2;
    }

    /* Kids: wander with water, flee nearby GPUs, walk to the well when drained, refill */
    for (int i = 0; i < N_KIDS; i++) {
        kid *k = &kids[i];
        int being_drunk = (gpus[0].state == GPU_DRINK && gpus[0].target == i) ||
                          (gpus[1].state == GPU_DRINK && gpus[1].target == i);
        double wvx = 0, wvy = 0;
        k->safe -= dt;
        if (being_drunk || k->state == KID_REFILL) {
            k->vx = k->vy = 0;
            k->since += dt;
            k->turn = 0;
            if (k->state == KID_REFILL) {
                k->timer -= dt;
                k->water = fmin(1, k->water + dt / 1.6);
                if (k->timer <= 0) {
                    k->state = KID_WATER;
                    k->water = 1;
                    k->safe = 2.5;
                    /* pick a spot away from the well */
                    double a = frand() * 2 * M_PI;
                    k->hx = WELL_X + cos(a) * (110 + frand() * 60);
                    k->hy = WELL_Y + 50 + sin(a) * 40;
                    clamp_arena(&k->hx, &k->hy, NULL, NULL);
                    k->homing = 1;
                }
            }
            continue;
        }
        if (k->state == KID_WATER) {
            if (kid_threat(k->x, k->y) > 0) {
                /*
                 * Run! Try 16 directions and take the one that ends up least threatened,
                 * which also slides along edges and out of corners. Favouring the way
                 * we already run keeps near-equal choices from flipping every frame.
                 */
                double v = hypot(k->vx, k->vy), best = 1e9, bx = 0, by = 0;
                for (int a = 0; a < 16; a++) {
                    double cx = cos(a * M_PI / 8), cy = sin(a * M_PI / 8);
                    double px = k->x + cx * 45, py = k->y + cy * 45;
                    clamp_arena(&px, &py, NULL, NULL);
                    double score = kid_threat(px, py) + (45 - hypot(px - k->x, py - k->y));
                    if (v > 10)
                        score -= 25 * (cx * k->vx + cy * k->vy) / v;
                    if (score < best) {
                        best = score;
                        bx = cx;
                        by = cy;
                    }
                }
                wvx = bx * 90;
                wvy = by * 90;
                /* once safe, keep strolling the same way, not back toward the GPU */
                k->dx = bx;
                k->dy = by * 0.5;
                k->timer = 1.5 + frand() * 1.5;
                k->homing = 0;                  /* escaped: stay out here, not back by the GPU */
            } else if (k->homing) {
                double dx = k->hx - k->x, dy = k->hy - k->y, d = hypot(dx, dy);
                if (d < 10) {
                    /* arrived: stroll on the same way for a bit, then wander from here */
                    double v = hypot(k->vx, k->vy);
                    k->homing = 0;
                    k->dx = v > 1 ? k->vx / v : 0;
                    k->dy = v > 1 ? k->vy / v * 0.5 : 0;
                    k->timer = 1 + frand() * 2;
                } else {
                    wvx = dx / d * 50;          /* head back out into town */
                    wvy = dy / d * 50;
                }
            } else {
                k->timer -= dt;
                if (k->timer <= 0) {
                    k->timer = 1.5 + frand() * 2.5;
                    double a = frand() * 2 * M_PI;
                    k->dx = cos(a);
                    k->dy = sin(a) * 0.5;
                }
                /* turn back at the edges */
                if ((k->x - ARENA_X0 < 8 && k->dx < 0) || (ARENA_X1 - k->x < 8 && k->dx > 0))
                    k->dx = -k->dx;
                if ((k->y - ARENA_Y0 < 4 && k->dy < 0) || (ARENA_Y1 - k->y < 4 && k->dy > 0))
                    k->dy = -k->dy;
                wvx = k->dx * 28;
                wvy = k->dy * 28;
            }
        } else {
            double dx = WELL_X + (i - 1.5) * 30 - k->x, dy = WELL_Y + 26 - k->y, d = hypot(dx, dy);
            if (d < 10) {
                k->state = KID_REFILL;
                k->timer = 1.6;
            } else {
                wvx = dx / d * 38 * fmin(1, d / 20);
                wvy = dy / d * 38 * fmin(1, d / 20);
            }
        }
        /* personal space */
        for (int j = 0; j < N_KIDS; j++) {
            if (j == i)
                continue;
            double dx = k->x - kids[j].x, dy = k->y - kids[j].y, d = hypot(dx, dy);
            if (d < 44 && d > 0.1) {
                wvx += dx / d * (44 - d) * 2;
                wvy += dy / d * (44 - d) * 2;
            }
        }
        steer(&k->x, &k->y, &k->vx, &k->vy, wvx, wvy, dt);
        update_face(&k->face, &k->turn, &k->since, k->vx, dt);
        double v = hypot(k->vx, k->vy);
        if (v > 3)
            k->step += dt * (3 + v * 0.12);
    }
    (void)t;
}

/* ---------------------------------------------------------------- render */

static void text_center(cairo_t *cr, double x, double y, double size, int bold, rgb c, const char *s)
{
    cairo_text_extents_t ext;
    cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL,
                           bold ? CAIRO_FONT_WEIGHT_BOLD : CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, size);
    cairo_text_extents(cr, s, &ext);
    double tx = x - ext.width / 2 - ext.x_bearing, ty = y - ext.height / 2 - ext.y_bearing;
    cairo_move_to(cr, tx, ty);
    cairo_text_path(cr, s);
    cairo_set_line_width(cr, size * 0.16);
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
    cairo_set_source_rgba(cr, 0.25, 0.12, 0.04, 0.85);
    cairo_stroke_preserve(cr);
    set_rgb(cr, c);
    cairo_fill(cr);
}

typedef struct { double zotac, tuf, tok; } shown_t;

static void draw_droplet(cairo_t *cr, double x, double y, double h, rgb c)
{
    cairo_move_to(cr, x, y - h / 2);
    cairo_curve_to(cr, x + h * 0.45, y, x + h * 0.4, y + h / 2, x, y + h / 2);
    cairo_curve_to(cr, x - h * 0.4, y + h / 2, x - h * 0.45, y, x, y - h / 2);
    set_rgb(cr, c);
    cairo_fill(cr);
}

static void draw_kid(cairo_t *cr, const kid *k, int being_drunk)
{
    rgb shirt = SHIRTS[k->shirt], skin = SKINS[k->skin];
    double leg = sin(k->step * 2) * 5;
    cairo_save(cr);
    cairo_translate(cr, k->x, k->y);
    cairo_scale(cr, k->face * KID_SCALE, KID_SCALE);
    /* shadow */
    cairo_save(cr);
    cairo_scale(cr, 1, 0.3);
    cairo_arc(cr, 0, 6, 10, 0, 2 * M_PI);
    cairo_restore(cr);
    cairo_set_source_rgba(cr, 0.3, 0.18, 0.08, 0.35);
    cairo_fill(cr);
    /* legs */
    cairo_set_line_width(cr, 3);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    cairo_set_source_rgb(cr, 0.25, 0.25, 0.35);
    cairo_move_to(cr, -2, -8); cairo_line_to(cr, -2 - leg, 0);
    cairo_move_to(cr, 2, -8); cairo_line_to(cr, 2 + leg, 0);
    cairo_stroke(cr);
    /* body */
    cairo_move_to(cr, -7, -8);
    cairo_line_to(cr, 7, -8);
    cairo_line_to(cr, 5, -22);
    cairo_line_to(cr, -5, -22);
    cairo_close_path(cr);
    set_rgb(cr, shirt);
    cairo_fill_preserve(cr);
    cairo_set_line_width(cr, 1);
    cairo_set_source_rgba(cr, 0.15, 0.08, 0.04, 0.8);
    cairo_stroke(cr);
    /* head */
    cairo_arc(cr, 0, -28, 7, 0, 2 * M_PI);
    set_rgb(cr, skin);
    cairo_fill_preserve(cr);
    cairo_set_source_rgba(cr, 0.15, 0.08, 0.04, 0.8);
    cairo_stroke(cr);
    cairo_arc(cr, 3, -29, 1.2, 0, 2 * M_PI);
    cairo_set_source_rgb(cr, 0.1, 0.1, 0.1);
    cairo_fill(cr);
    /* mouth: worried when a GPU is at the cup, a smile otherwise */
    cairo_set_line_width(cr, 1.2);
    if (being_drunk || k->state == KID_DRAINED)
        cairo_arc(cr, 3, -23.5, 2, M_PI * 1.15, M_PI * 1.85);
    else
        cairo_arc(cr, 3, -26.5, 2, M_PI * 0.15, M_PI * 0.85);
    cairo_stroke(cr);
    /* arm and cup held out in front */
    cairo_set_line_width(cr, 2.5);
    set_rgb(cr, skin);
    cairo_move_to(cr, 3, -18); cairo_line_to(cr, 11, -15);
    cairo_stroke(cr);
    cairo_rectangle(cr, 10, -22, 8, 10);
    cairo_set_source_rgb(cr, 0.92, 0.94, 0.98);
    cairo_fill(cr);
    if (k->water > 0.02) {
        double h = 9 * k->water;
        cairo_rectangle(cr, 10.5, -12.5 - h, 7, h);
        set_rgb(cr, WATER);
        cairo_fill(cr);
    }
    cairo_restore(cr);
}

static void draw_gpu(cairo_t *cr, const gpu *g, int gi, double t)
{
    rgb body = GPU_COL[gi], dark = lerp(body, (rgb){ 0, 0, 0 }, 0.5);
    double leg = g->state == GPU_NAP ? 0 : sin(g->step * 2) * 4;
    cairo_save(cr);
    cairo_translate(cr, g->x, g->y);
    cairo_scale(cr, g->face * GPU_SCALE, GPU_SCALE);
    /* shadow */
    cairo_save(cr);
    cairo_scale(cr, 1, 0.25);
    cairo_arc(cr, 0, 4, 34, 0, 2 * M_PI);
    cairo_restore(cr);
    cairo_set_source_rgba(cr, 0.3, 0.18, 0.08, 0.35);
    cairo_fill(cr);
    /* stubby legs */
    cairo_set_line_width(cr, 3.5);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    cairo_set_source_rgb(cr, 0.15, 0.15, 0.18);
    for (int l = 0; l < 4; l++) {
        double lx = -24 + l * 16, sw = (l & 1 ? leg : -leg);
        cairo_move_to(cr, lx, -6);
        cairo_line_to(cr, lx + sw, 0);
    }
    cairo_stroke(cr);
    /* card body with a PCB edge and gold fingers */
    cairo_rectangle(cr, -34, -34, 68, 28);
    set_rgb(cr, body);
    cairo_fill(cr);
    cairo_rectangle(cr, -34, -8, 68, 3);
    cairo_set_source_rgb(cr, 0.10, 0.35, 0.18);
    cairo_fill(cr);
    cairo_rectangle(cr, -20, -5, 26, 2);
    cairo_set_source_rgb(cr, 0.95, 0.78, 0.25);
    cairo_fill(cr);
    cairo_rectangle(cr, -34, -34, 68, 28);
    cairo_set_line_width(cr, 2);
    cairo_set_source_rgb(cr, 0.10, 0.08, 0.10);
    cairo_stroke(cr);
    (void)dark;
    /* two spinning fans */
    for (int f = 0; f < 2; f++) {
        double fx = -15 + f * 30, fy = -20;
        cairo_arc(cr, fx, fy, 11, 0, 2 * M_PI);
        cairo_set_source_rgb(cr, 0.12, 0.12, 0.15);
        cairo_fill(cr);
        cairo_set_source_rgba(cr, 0.6, 0.62, 0.7, 0.9);
        for (int b = 0; b < 5; b++) {
            double a = g->fan * 2 * M_PI + b * 2 * M_PI / 5 + f;
            cairo_move_to(cr, fx, fy);
            cairo_arc(cr, fx, fy, 9.5, a, a + 0.55);
            cairo_close_path(cr);
        }
        cairo_fill(cr);
    }
    /* LED eyes on the front edge */
    rgb led = g->state == GPU_NAP ? (rgb){ 0.3, 0.3, 0.35 } : (rgb){ 1.0, 0.25, 0.2 };
    set_rgb(cr, led);
    cairo_rectangle(cr, 26, -30, 5, 3);
    cairo_rectangle(cr, 26, -24, 5, 3);
    cairo_fill(cr);

    /* The straw: up and forward; when drinking it reaches the cup with water flowing up it */
    double sx0 = 20, sy0 = -34;
    double sx1 = g->state == GPU_DRINK ? 40 : 34, sy1 = g->state == GPU_DRINK ? -18 : -58;
    cairo_move_to(cr, sx0, sy0);
    cairo_curve_to(cr, sx0 + 2, sy0 - 18, sx1 - 6, sy1 - 12, sx1, sy1);
    cairo_set_line_width(cr, 4);
    cairo_set_source_rgb(cr, 0.95, 0.95, 0.95);
    cairo_stroke_preserve(cr);
    double dashes[2] = { 4, 4 };
    cairo_set_dash(cr, dashes, 2, g->state == GPU_DRINK ? -t * 30 : 0);
    cairo_set_line_width(cr, 3);
    if (g->state == GPU_DRINK)
        set_rgb(cr, WATER);
    else
        cairo_set_source_rgb(cr, 0.9, 0.2, 0.25);
    cairo_stroke(cr);
    cairo_set_dash(cr, NULL, 0, 0);
    cairo_restore(cr);

    if (g->state == GPU_DRINK) {
        cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
        cairo_set_font_size(cr, 20 + 3 * sin(t * 12));
        cairo_move_to(cr, g->x - 30, g->y - 108);
        cairo_set_source_rgb(cr, 0.15, 0.35, 0.8);
        cairo_show_text(cr, "SLURP");
    } else if (g->state == GPU_NAP) {
        cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
        double ph = fmod(t * 0.5 + gi * 0.5, 1.0);
        cairo_set_font_size(cr, 16 + ph * 12);
        cairo_move_to(cr, g->x + 20 + ph * 16, g->y - 56 - ph * 30);
        cairo_set_source_rgba(cr, 0.3, 0.3, 0.4, 1 - ph);
        cairo_show_text(cr, "z");
    }
}

static void update_hud(const stats *s, const shown_t *sh, double t)
{
    const double c = SIZE / 2.0;
    static double tok, watts, lit, next;
    char key[128], txt[64];
    if (t >= next || t < next - 1) {
        next = t + 0.25;
        tok = sh->tok;
        watts = s->power[0] + s->power[1];
        lit = liters;
    }
    snprintf(key, sizeof(key), "%.1f|%.0f|%.0f|%d", lit, tok, watts, s->tok_s < 1 && s->running == 0);
    if (!strcmp(key, hud_key))
        return;
    strcpy(hud_key, key);
    cairo_t *cr = cairo_create(hud_cache);
    cairo_set_operator(cr, CAIRO_OPERATOR_CLEAR);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);

    snprintf(txt, sizeof(txt), lit < 100 ? "%.1f L" : "%.0f L", lit);
    text_center(cr, c + 16, 58, 46, 1, (rgb){ 1, 1, 1 }, txt);
    draw_droplet(cr, c - 72 - (lit >= 10 ? 14 : 0), 58, 38, WATER);
    text_center(cr, c, 100, 16, 1, (rgb){ 1, 0.95, 0.85 }, "datacenter water");

    if (s->tok_s < 1 && s->running == 0)
        snprintf(txt, sizeof(txt), "idle  \xC2\xB7  %.0f W", watts);
    else
        snprintf(txt, sizeof(txt), "%.0f tok/s \xC2\xB7 %.0f W", tok, watts);
    text_center(cr, c, 446, 22, 1, (rgb){ 1, 1, 1 }, txt);
    cairo_destroy(cr);
}

static void render(cairo_t *cr, const stats *s, const shown_t *sh, double t)
{
    cairo_set_source_surface(cr, bg_cache, 0, 0);
    cairo_paint(cr);

    cairo_save(cr);
    cairo_translate(cr, 0, BG_SHIFT);
    /* Water tower tank with the level dropping */
    cairo_rectangle(cr, 356, 214, 72, 54);
    cairo_set_source_rgb(cr, 0.55, 0.40, 0.30);
    cairo_fill(cr);
    double lvl = 50 * tank;
    cairo_rectangle(cr, 360, 264 - lvl, 64, lvl);
    set_rgb(cr, lerp((rgb){ 0.35, 0.25, 0.15 }, WATER, clamp01(tank * 3)));
    cairo_fill(cr);
    cairo_rectangle(cr, 356, 214, 72, 54);
    cairo_set_line_width(cr, 2);
    cairo_set_source_rgb(cr, 0.35, 0.25, 0.18);
    cairo_stroke(cr);
    cairo_move_to(cr, 350, 214);
    cairo_line_to(cr, 392, 196);
    cairo_line_to(cr, 434, 214);
    cairo_close_path(cr);
    cairo_set_source_rgb(cr, 0.45, 0.32, 0.24);
    cairo_fill(cr);

    /* Blinking server lights in the datacenter windows, faster when busy */
    for (int i = 0; i < 12; i++) {
        int on = ((int)(t * (s->tok_s > 1 ? 9 : 1.5)) + i * 7) % 3 != 0;
        cairo_rectangle(cr, 118 + (i % 4) * 14, 232 + (i / 4) * 9, 8, 4);
        if (on)
            cairo_set_source_rgb(cr, 0.3, 1.0, 0.5);
        else
            cairo_set_source_rgb(cr, 0.2, 0.25, 0.25);
        cairo_fill(cr);
    }

    cairo_restore(cr);

    /* Draw everyone back to front by depth */
    int order[N_KIDS + 2];
    double depth[N_KIDS + 2];
    for (int i = 0; i < N_KIDS; i++) {
        order[i] = i;
        depth[i] = kids[i].y;
    }
    for (int g = 0; g < 2; g++) {
        order[N_KIDS + g] = N_KIDS + g;
        depth[N_KIDS + g] = gpus[g].y;
    }
    for (int i = 1; i < N_KIDS + 2; i++)
        for (int j = i; j > 0 && depth[order[j]] < depth[order[j - 1]]; j--) {
            int tmp = order[j]; order[j] = order[j - 1]; order[j - 1] = tmp;
        }
    for (int i = 0; i < N_KIDS + 2; i++) {
        int o = order[i];
        if (o < N_KIDS) {
            int drunk = (gpus[0].state == GPU_DRINK && gpus[0].target == o) ||
                        (gpus[1].state == GPU_DRINK && gpus[1].target == o);
            draw_kid(cr, &kids[o], drunk);
        } else {
            draw_gpu(cr, &gpus[o - N_KIDS], o - N_KIDS, t);
        }
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
    build_background();
    init_actors();

    surf = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cr = cairo_create(surf);
    tj = tj3Init(TJINIT_COMPRESS);
    tj3Set(tj, TJPARAM_QUALITY, JPEG_QUALITY);
    tj3Set(tj, TJPARAM_SUBSAMP, TJSAMP_420);

    if (bench) {
        struct { double tok0, tok1; double w0, w1; double run_s; const char *png; } scenes[] = {
            { 723, 884, 470, 460, 12, "thirst_preview.png" },
            { 180, 60, 250, 120, 12, "thirst_mid.png" },
            { 0, 0, 25, 30, 8, "thirst_idle.png" },
        };
        for (int k = 0; k < 3; k++) {
            init_actors();
            liters = k == 2 ? 3.2 : 0;
            tank = 1;
            s.tok_port[0] = scenes[k].tok0; s.tok_port[1] = scenes[k].tok1;
            s.tok_s = s.tok_port[0] + s.tok_port[1]; s.running = s.tok_s > 0 ? 4 : 0;
            s.power[0] = scenes[k].w0; s.power[1] = scenes[k].w1; s.temp[0] = 54; s.temp[1] = 71;
            sh.tok = s.tok_s;
            int n = (int)(scenes[k].run_s * FPS_BUSY);
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
            printf("%s: %.2f ms/frame, jpeg %zu bytes, %.2f L\n", scenes[k].png, (now_s() - b0) * 1000 / 60, len, liters);
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

        int idle_now = s.tok_s < 1 && s.running == 0;
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
    cairo_surface_destroy(bg_cache);
    cairo_surface_destroy(hud_cache);
    if (nvml_ok)
        nvmlShutdown();
    if (fd >= 0)
        close(fd);
    return 0;
}
