/*
 * tears2: a painted version of tears on the iCUE LINK AIO pump LCD.
 *
 * The bedroom, the GPU robot and the child (in four matching moods: content,
 * worried, sad, sobbing) are generated art in assets/tears2/. The child crossfades
 * between moods as sustained token load makes them sadder (~60 s memory), and
 * recovers when idle; the room is washed colder as it goes. Animated tears run from
 * the painted eye down the cheek at the token rate, and the robot's arm raises a
 * glass to the jaw to catch them, slurping it dry when full.
 * The mL counter uses the same datacenter-water estimate as thirst (ML_PER_TOKEN).
 * Run with --demo to simulate data, --showcase for a scripted 36 s story that ends
 * sobbing (for filming and GIFs), --bench to write preview PNGs.
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
#define MAX_TEARS_PER_S 7.0
#define TEARS_PER_GLASS 22.0
#define SADNESS_TAU     60.0        /* seconds: how long load lingers in the child's mood */
#define SADNESS_FULL    40000.0     /* recent tokens at which the child is inconsolable */
#define MAX_TEARS       80
#define N_MOODS         4

/* Where the painted child sits on screen (the 430 px sprite) */
#define CHILD_X         (-40.0)
#define CHILD_Y         20.0
/* Where the painted robot sits (195 px wide, mirrored so its arm socket faces the child) */
#define ROBOT_X         300.0
#define ROBOT_Y         178.0
#define SHOULDER_X      320.0
#define SHOULDER_Y      306.0
#define ARM_A           70.0
#define ARM_B           66.0
/* Glass: held under the jaw, or lowered by the robot */
#define GLASS_UP_X      274.0
#define GLASS_UP_Y      348.0
#define GLASS_DOWN_X    330.0
#define GLASS_DOWN_Y    430.0

typedef struct { double x, y, vy, size, u, grow; int phase, alive; } tear;

static tear   tears[MAX_TEARS];
static double tear_acc, glass, recent, sadness, harvested_ml, arm_p, slurp_t = -1;
/* Shown mood: one painting at a time, switching with a short crossfade */
#define MOOD_FADE_S     0.5
static const double mood_at[N_MOODS] = { 0.0, 0.2, 0.5, 0.8 };     /* sadness where each mood starts */
static int    mood = 0, prev_mood = 0;
static double mood_fade = 1;                                        /* 0 just switched .. 1 done */
static cairo_surface_t *room_img, *robot_img, *child_img[N_MOODS], *hud_cache;
static char hud_key[128];

static double frand(void) { return rand() / (double)RAND_MAX; }
static double smooth(double x) { x = clamp01(x); return x * x * (3 - 2 * x); }

static const rgb TEAR = { 0.70, 0.88, 1.00 };

/* The tear's path on screen: from the lower lid of the painted right eye, down the cheek, off the jaw */
static void cheek_path(double u, double *x, double *y)
{
    static const double p[][2] = { { 254, 261 }, { 261, 275 }, { 266, 289 }, { 268, 302 }, { 270, 314 } };
    double f = clamp01(u) * 4;
    int i = (int)f;
    if (i >= 4) { *x = p[4][0]; *y = p[4][1]; return; }
    double t = f - i;
    *x = p[i][0] + (p[i + 1][0] - p[i][0]) * t;
    *y = p[i][1] + (p[i + 1][1] - p[i][1]) * t;
}

static void glass_pos(double *x, double *y)
{
    double k = smooth(arm_p);
    *x = GLASS_DOWN_X + (GLASS_UP_X - GLASS_DOWN_X) * k;
    *y = GLASS_DOWN_Y + (GLASS_UP_Y - GLASS_DOWN_Y) * k;
}

/* Assets live next to the binary (assets/tears2/), or in ./assets/tears2 when run from the repo */
static cairo_surface_t *load_asset(const char *name)
{
    char exe[512], path[1024];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n > 0) {
        exe[n] = 0;
        char *slash = strrchr(exe, '/');
        if (slash)
            *slash = 0;
        snprintf(path, sizeof(path), "%s/assets/tears2/%s", exe, name);
        cairo_surface_t *s = cairo_image_surface_create_from_png(path);
        if (cairo_surface_status(s) == CAIRO_STATUS_SUCCESS)
            return s;
        cairo_surface_destroy(s);
    }
    snprintf(path, sizeof(path), "assets/tears2/%s", name);
    cairo_surface_t *s = cairo_image_surface_create_from_png(path);
    if (cairo_surface_status(s) != CAIRO_STATUS_SUCCESS) {
        fprintf(stderr, "tears2: can't load %s (looked next to the binary and in ./assets/tears2)\n", name);
        exit(1);
    }
    return s;
}

static void load_assets(void)
{
    static const char *moods[N_MOODS] = { "child_0_content.png", "child_1_worried.png", "child_2_sad.png", "child_3_crying.png" };
    room_img = load_asset("room.png");
    robot_img = load_asset("robot.png");
    for (int i = 0; i < N_MOODS; i++)
        child_img[i] = load_asset(moods[i]);
    hud_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, SIZE);
}

/* ---------------------------------------------------------------- simulation */

static double scripted_sadness = -1;    /* set by --showcase: overrides the mood memory */

static void simulate(const stats *s, double dt)
{
    int idle = s->tok_s < 1 && s->running == 0;

    recent = recent * exp(-dt / SADNESS_TAU) + s->tok_s * dt;
    if (scripted_sadness >= 0)
        sadness = scripted_sadness;
    else
        sadness += (clamp01(recent / SADNESS_FULL) - sadness) * fmin(1, dt * 0.8);
    harvested_ml += s->tok_s * dt * ML_PER_TOKEN;

    /* Switch paintings when sadness clearly crosses a mood boundary */
    int want_mood = mood;
    while (want_mood < N_MOODS - 1 && sadness >= mood_at[want_mood + 1] + 0.03)
        want_mood++;
    while (want_mood > 0 && sadness < mood_at[want_mood] - 0.03)
        want_mood--;
    if (want_mood != mood) {
        prev_mood = mood;
        mood = want_mood;
        mood_fade = 0;
    }
    mood_fade = fmin(1, mood_fade + dt / MOOD_FADE_S);

    /* The arm raises the glass while tears flow, lowers it to slurp or when idle */
    if (slurp_t < 0 && glass >= 1 && arm_p > 0.95)
        slurp_t = 0;
    double want = idle ? 0 : 1;
    if (slurp_t >= 0) {
        want = 0.2;
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

    tear_acc += (idle ? 0 : fmin(s->tok_s / TOKENS_PER_TEAR, MAX_TEARS_PER_S)) * dt;
    while (tear_acc >= 1) {
        for (int i = 0; i < MAX_TEARS; i++)
            if (!tears[i].alive) {
                tears[i] = (tear){ 0, 0, 0, 3.2 + frand() * 1.4 + 1.5 * sadness, 0, 0, 0, 1 };
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
        if (d->phase == 0) {                        /* welling on the lower lid */
            d->grow += dt * (2.5 + 2 * sadness);
            cheek_path(0, &d->x, &d->y);
            if (d->grow >= 1)
                d->phase = 1;
        } else if (d->phase == 1) {                 /* rolling down the cheek */
            d->u += dt * (0.7 + d->u * 1.6);
            cheek_path(d->u, &d->x, &d->y);
            if (d->u >= 1) {
                d->phase = 2;
                d->vy = 30;
            }
        } else {                                    /* falling from the jaw */
            d->vy += 600 * dt;
            d->y += d->vy * dt;
            d->x += 6 * dt;
            if (arm_p > 0.9 && slurp_t < 0 && d->y >= gy - 20 && fabs(d->x - gx) < 16) {
                d->alive = 0;
                glass = fmin(1, glass + 1 / TEARS_PER_GLASS);
            } else if (d->y > 470) {
                d->alive = 0;
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
    cairo_set_source_rgba(cr, 0.10, 0.08, 0.14, 0.8);
    cairo_stroke_preserve(cr);
    set_rgb(cr, c);
    cairo_fill(cr);
}

typedef struct { double zotac, tuf, tok; } shown_t;

static void draw_tear(cairo_t *cr, const tear *d)
{
    double r = d->size * (d->phase == 0 ? 0.4 + 0.6 * d->grow : 1);
    double stretch = d->phase == 2 ? 1.3 : 1.05;
    cairo_save(cr);
    cairo_translate(cr, d->x, d->y);
    cairo_move_to(cr, 0, -r * 1.7 * stretch);
    cairo_curve_to(cr, r, -r * 0.3, r * 1.05, r, 0, r);
    cairo_curve_to(cr, -r * 1.05, r, -r, -r * 0.3, 0, -r * 1.7 * stretch);
    cairo_pattern_t *g = cairo_pattern_create_radial(-r * 0.3, -r * 0.3, 0, 0, 0, r * 1.4);
    cairo_pattern_add_color_stop_rgba(g, 0, 1, 1, 1, 0.95);
    cairo_pattern_add_color_stop_rgba(g, 0.4, TEAR.r, TEAR.g, TEAR.b, 0.8);
    cairo_pattern_add_color_stop_rgba(g, 1, 0.35, 0.60, 0.90, 0.85);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    cairo_restore(cr);
}

/* The robot's arm: dark steel segments with brass joints, to match the painting */
static void draw_arm(cairo_t *cr, double hx, double hy)
{
    double dx = hx - SHOULDER_X, dy = hy - SHOULDER_Y, d = fmin(hypot(dx, dy), ARM_A + ARM_B - 1);
    double base = atan2(dy, dx);
    double a = acos(fmax(-1, fmin(1, (ARM_A * ARM_A + d * d - ARM_B * ARM_B) / (2 * ARM_A * d))));
    double ex = SHOULDER_X + ARM_A * cos(base + a), ey = SHOULDER_Y + ARM_A * sin(base + a);   /* elbow down */
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
    cairo_move_to(cr, SHOULDER_X, SHOULDER_Y);
    cairo_line_to(cr, ex, ey);
    cairo_line_to(cr, hx, hy);
    cairo_set_line_width(cr, 13);
    cairo_set_source_rgb(cr, 0.10, 0.09, 0.09);
    cairo_stroke_preserve(cr);
    cairo_set_line_width(cr, 8);
    cairo_set_source_rgb(cr, 0.36, 0.35, 0.35);
    cairo_stroke_preserve(cr);
    cairo_set_line_width(cr, 2.5);
    cairo_set_source_rgba(cr, 0.75, 0.72, 0.68, 0.7);
    cairo_stroke(cr);
    double jx[3] = { SHOULDER_X, ex, hx }, jy[3] = { SHOULDER_Y, ey, hy };
    for (int j = 0; j < 3; j++) {
        cairo_arc(cr, jx[j], jy[j], j == 2 ? 6 : 7.5, 0, 2 * M_PI);
        cairo_pattern_t *g = cairo_pattern_create_radial(jx[j] - 2, jy[j] - 2, 1, jx[j], jy[j], 8);
        cairo_pattern_add_color_stop_rgb(g, 0, 0.95, 0.78, 0.45);
        cairo_pattern_add_color_stop_rgb(g, 1, 0.55, 0.40, 0.20);
        cairo_set_source(cr, g);
        cairo_fill(cr);
        cairo_pattern_destroy(g);
    }
}

static void draw_glass(cairo_t *cr, double gx, double gy)
{
    const double wt = 34, wb = 26, h = 44;
    double y0 = gy - h / 2, y1 = gy + h / 2;
    if (glass > 0.01) {
        double lh = (h - 6) * glass, yt = y1 - 3 - lh, w = wb + (wt - wb) * ((lh + 3) / h);
        cairo_move_to(cr, gx - w / 2 + 2, yt);
        cairo_line_to(cr, gx + w / 2 - 2, yt);
        cairo_line_to(cr, gx + wb / 2 - 2, y1 - 3);
        cairo_line_to(cr, gx - wb / 2 + 2, y1 - 3);
        cairo_close_path(cr);
        cairo_pattern_t *g = cairo_pattern_create_linear(0, yt, 0, y1);
        cairo_pattern_add_color_stop_rgba(g, 0, 0.80, 0.92, 1.0, 0.9);
        cairo_pattern_add_color_stop_rgba(g, 1, 0.40, 0.64, 0.95, 0.9);
        cairo_set_source(cr, g);
        cairo_fill(cr);
        cairo_pattern_destroy(g);
    }
    cairo_move_to(cr, gx - wt / 2, y0);
    cairo_line_to(cr, gx + wt / 2, y0);
    cairo_line_to(cr, gx + wb / 2, y1);
    cairo_line_to(cr, gx - wb / 2, y1);
    cairo_close_path(cr);
    cairo_set_source_rgba(cr, 0.90, 0.96, 1.0, 0.22);
    cairo_fill_preserve(cr);
    cairo_set_line_width(cr, 2);
    cairo_set_source_rgba(cr, 1, 1, 1, 0.9);
    cairo_stroke(cr);
    cairo_move_to(cr, gx - wt / 2 + 5, y0 + 6);
    cairo_line_to(cr, gx - wb / 2 + 4, y1 - 7);
    cairo_set_line_width(cr, 2.5);
    cairo_set_source_rgba(cr, 1, 1, 1, 0.55);
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
    soft_text(cr, c, 44, 28, (rgb){ 0.80, 0.93, 1.0 }, txt);
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

    /* The painted bedroom, washed colder and darker as the child gets sadder */
    cairo_set_source_surface(cr, room_img, 0, 0);
    cairo_paint(cr);
    cairo_set_source_rgba(cr, 0.10, 0.16, 0.34, 0.55 * sadness);
    cairo_paint(cr);

    cairo_set_source_surface(cr, robot_img, ROBOT_X, ROBOT_Y);
    cairo_paint(cr);

    /* The child: one painted mood at a time, with a quick crossfade when it changes */
    if (mood_fade < 1) {
        cairo_set_source_surface(cr, child_img[prev_mood], CHILD_X, CHILD_Y);
        cairo_paint(cr);
        cairo_set_source_surface(cr, child_img[mood], CHILD_X, CHILD_Y);
        cairo_paint_with_alpha(cr, smooth(mood_fade));
    } else {
        cairo_set_source_surface(cr, child_img[mood], CHILD_X, CHILD_Y);
        cairo_paint(cr);
    }

    for (int i = 0; i < MAX_TEARS; i++)
        if (tears[i].alive)
            draw_tear(cr, &tears[i]);

    double gx, gy;
    glass_pos(&gx, &gy);
    draw_glass(cr, gx, gy);
    draw_arm(cr, gx + 19, gy + 6);

    /* Slurping through a bendy straw into the robot's side vent */
    if (slurp_t >= 0 && arm_p < 0.35) {
        cairo_move_to(cr, gx + 4, gy - 10);
        cairo_curve_to(cr, gx + 4, gy - 80, 330, 290, 352, 300);
        cairo_set_line_width(cr, 6);
        cairo_set_source_rgb(cr, 1, 1, 1);
        cairo_stroke_preserve(cr);
        double dashes[2] = { 6, 6 };
        cairo_set_dash(cr, dashes, 2, -t * 60);
        cairo_set_line_width(cr, 4.5);
        set_rgb(cr, TEAR);
        cairo_stroke(cr);
        cairo_set_dash(cr, NULL, 0, 0);
        soft_text(cr, 400, 168, 24 + 3 * sin(t * 14), (rgb){ 0.80, 0.93, 1.0 }, "SLURP");
    }

    /* round vignette */
    cairo_pattern_t *g = cairo_pattern_create_radial(c, c, 185, c, c, 250);
    cairo_pattern_add_color_stop_rgba(g, 0, 0, 0, 0, 0);
    cairo_pattern_add_color_stop_rgba(g, 1, 0, 0, 0, 0.6);
    cairo_set_source(cr, g);
    cairo_paint(cr);
    cairo_pattern_destroy(g);

    update_hud(s, sh, t);
    cairo_set_source_surface(cr, hud_cache, 0, 0);
    cairo_paint(cr);
}

/* ---------------------------------------------------------------- showcase */

/*
 * --showcase: a scripted 36 s story for filming or GIFs. The mood would take minutes of
 * sustained load to go all the way; here it is driven directly: content, then steadily
 * sadder under rising load, ending on sobbing and holding there.
 */
#define SHOWCASE_LEN    36.0

static void showcase_poll(stats *s, double t)
{
    double lt = fmod(t, SHOWCASE_LEN), tok, sad;
    if (lt < 4) {
        tok = 180;
        sad = 0;
    } else if (lt < 28) {
        double u = (lt - 4) / 24;
        tok = 250 + 1450 * u;
        sad = u * u * (3 - 2 * u);
    } else {
        tok = 1750;
        sad = 1;
    }
    scripted_sadness = sad;
    s->tok_port[0] = tok * 0.46;
    s->tok_port[1] = tok * 0.54;
    s->tok_s = tok;
    s->running = 4;
    s->power[0] = 30 + fmin(545, tok * 0.28);
    s->power[1] = 35 + fmin(540, tok * 0.29);
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

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--demo"))
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
    load_assets();

    surf = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cr = cairo_create(surf);
    tj = tj3Init(TJINIT_COMPRESS);
    tj3Set(tj, TJPARAM_QUALITY, JPEG_QUALITY);
    tj3Set(tj, TJPARAM_SUBSAMP, TJSAMP_420);

    if (bench) {
        struct { double tok; double secs; const char *png; } scenes[] = {
            { 150, 20, "tears2_content.png" },
            { 450, 40, "tears2_worried.png" },
            { 900, 50, "tears2_sad.png" },
            { 1600, 90, "tears2_preview.png" },
        };
        for (int k = 0; k < 4; k++) {
            memset(tears, 0, sizeof(tears));
            recent = sadness = glass = harvested_ml = arm_p = 0;
            slurp_t = -1;
            mood = prev_mood = 0;
            mood_fade = 1;
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
                simulate(&s, 1.0 / FPS_BUSY);
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
                vllm_poll(&s, t);
            }
        }

        k = dt * 4 > 1 ? 1 : dt * 4;
        sh.tok += (s.tok_s - sh.tok) * k;

        simulate(&s, dt);
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
    cairo_surface_destroy(room_img);
    cairo_surface_destroy(robot_img);
    for (int i = 0; i < N_MOODS; i++)
        cairo_surface_destroy(child_img[i]);
    cairo_surface_destroy(hud_cache);
    if (nvml_ok)
        nvmlShutdown();
    if (fd >= 0)
        close(fd);
    return 0;
}
