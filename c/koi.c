/*
 * koi: a koi pond on the iCUE LINK AIO pump LCD, fed by your LLM.
 *
 * The pond painting and the two koi sprites were generated with an image model
 * (assets/koi/). Blue koi belong to GPU 0's vLLM server, orange-and-white koi to
 * GPU 1's. Each server's tokens drop food pellets on the water with a ripple in its
 * colour; its koi chase and eat them. The sprites are cut into strips and waved so
 * the fish swim rather than slide. Idle, the koi drift slowly.
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

/* ---------------------------------------------------------------- pond */

#define FPS_BUSY        20
#define FPS_IDLE        10
#define N_KOI           6           /* 0..2 blue (GPU 0's server), 3..5 orange (GPU 1's) */
#define KOI_LEN         132.0       /* on-screen length of a full-size koi */
#define POND_R          150.0       /* koi stay inside this radius (the water, not the rocks) */
#define TOKENS_PER_PELLET 24.0
#define MAX_PELLETS_PER_S 3.0       /* per server */
#define MAX_PELLETS     160
#define MAX_RIPPLES     80
#define SLICES          10          /* strips the sprite is cut into so the body can undulate */

typedef struct {
    double x, y, heading, speed, turn, phase, size;
    int    src;
} koi;

typedef struct { double x, y, age; int src, alive; } pellet;
typedef struct { double x, y, age, max_r; rgb col; int alive; } ripple;

static koi     kois[N_KOI];
static pellet  pellets[MAX_PELLETS];
static ripple  ripples[MAX_RIPPLES];
static double  pellet_acc[2];
static cairo_surface_t *pond_img, *koi_img[2], *hud_cache;
static char hud_key[128];

static double frand(void) { return rand() / (double)RAND_MAX; }

static const rgb KOI_TINT[2] = { { 0.55, 0.75, 1.0 }, { 1.0, 0.62, 0.30 } };

/* Assets live next to the binary (assets/koi/), or in ./assets/koi when run from the repo */
static cairo_surface_t *load_asset(const char *name)
{
    char exe[512], path[1024];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n > 0) {
        exe[n] = 0;
        char *slash = strrchr(exe, '/');
        if (slash)
            *slash = 0;
        snprintf(path, sizeof(path), "%s/assets/koi/%s", exe, name);
        cairo_surface_t *s = cairo_image_surface_create_from_png(path);
        if (cairo_surface_status(s) == CAIRO_STATUS_SUCCESS)
            return s;
        cairo_surface_destroy(s);
    }
    snprintf(path, sizeof(path), "assets/koi/%s", name);
    cairo_surface_t *s = cairo_image_surface_create_from_png(path);
    if (cairo_surface_status(s) != CAIRO_STATUS_SUCCESS) {
        fprintf(stderr, "koi: can't load %s (looked next to the binary and in ./assets/koi)\n", name);
        exit(1);
    }
    return s;
}

static void load_assets(void)
{
    pond_img = load_asset("pond.png");
    const char *names[2] = { "koi_blue.png", "koi_orange.png" };
    for (int i = 0; i < 2; i++) {
        /* Scale once to the largest on-screen size */
        cairo_surface_t *src = load_asset(names[i]);
        double sw = cairo_image_surface_get_width(src), shh = cairo_image_surface_get_height(src);
        double k = KOI_LEN * 1.05 / shh;
        int w = (int)ceil(sw * k), h = (int)ceil(shh * k);
        koi_img[i] = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w, h);
        cairo_t *c = cairo_create(koi_img[i]);
        cairo_scale(c, k, k);
        cairo_set_source_surface(c, src, 0, 0);
        cairo_pattern_set_filter(cairo_get_source(c), CAIRO_FILTER_BEST);
        cairo_paint(c);
        cairo_destroy(c);
        cairo_surface_destroy(src);
    }
    hud_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, SIZE);
}

static void init_koi(void)
{
    static const double sizes[N_KOI] = { 1.0, 0.85, 0.72, 1.05, 0.88, 0.75 };
    for (int i = 0; i < N_KOI; i++) {
        koi *k = &kois[i];
        double a = frand() * 2 * M_PI, r = 40 + frand() * 90;
        k->x = SIZE / 2.0 + cos(a) * r;
        k->y = SIZE / 2.0 + sin(a) * r;
        k->heading = frand() * 2 * M_PI;
        k->speed = 20;
        k->phase = frand() * 6;
        k->size = sizes[i];
        k->src = i < 3 ? 0 : 1;
    }
}

static void add_ripple(double x, double y, double max_r, rgb col)
{
    for (int i = 0; i < MAX_RIPPLES; i++)
        if (!ripples[i].alive) {
            ripples[i] = (ripple){ x, y, 0, max_r, col, 1 };
            return;
        }
}

/* ---------------------------------------------------------------- simulation */

static int nearest_pellet(const koi *k)
{
    int best = -1;
    double bd = 260;
    for (int i = 0; i < MAX_PELLETS; i++) {
        if (!pellets[i].alive || pellets[i].src != k->src)
            continue;
        double d = hypot(pellets[i].x - k->x, pellets[i].y - k->y);
        if (d < bd) {
            bd = d;
            best = i;
        }
    }
    return best;
}

static void simulate(const stats *s, double dt)
{
    const double c = SIZE / 2.0;
    int idle = s->tok_s < 1 && s->running == 0;

    /* Food: each server's tokens drop pellets on the water, with a ripple where they land */
    for (int src = 0; src < 2; src++) {
        pellet_acc[src] += fmin(s->tok_port[src] / TOKENS_PER_PELLET, MAX_PELLETS_PER_S) * dt;
        while (pellet_acc[src] >= 1) {
            for (int i = 0; i < MAX_PELLETS; i++)
                if (!pellets[i].alive) {
                    double a = frand() * 2 * M_PI, r = sqrt(frand()) * (POND_R - 10);
                    pellets[i] = (pellet){ c + cos(a) * r, c + sin(a) * r, 0, src, 1 };
                    add_ripple(pellets[i].x, pellets[i].y, 26, KOI_TINT[src]);
                    break;
                }
            pellet_acc[src] -= 1;
        }
    }
    for (int i = 0; i < MAX_PELLETS; i++)
        if (pellets[i].alive && (pellets[i].age += dt) > 10)
            pellets[i].alive = 0;           /* sank */

    for (int i = 0; i < N_KOI; i++) {
        koi *k = &kois[i];
        int p = nearest_pellet(k);
        double want_speed = idle ? 14 : 26;
        k->turn += (frand() - 0.5) * 1.6 * dt;
        k->turn *= exp(-dt * 1.0);

        if (p >= 0) {
            /* chase the food */
            double to = atan2(pellets[p].y - k->y, pellets[p].x - k->x);
            k->turn += remainder(to - k->heading, 2 * M_PI) * 2.5 * dt;
            want_speed = 55 + 25 * k->size;
            double mx = k->x + cos(k->heading) * KOI_LEN * 0.45 * k->size;
            double my = k->y + sin(k->heading) * KOI_LEN * 0.45 * k->size;
            if (hypot(pellets[p].x - mx, pellets[p].y - my) < 12) {
                pellets[p].alive = 0;
                add_ripple(mx, my, 18, (rgb){ 1, 1, 1 });
            }
        }
        /* stay in the pond */
        double dx = k->x - c, dy = k->y - c, r = hypot(dx, dy);
        if (r > POND_R - 40) {
            double to_center = atan2(-dy, -dx);
            k->turn += remainder(to_center - k->heading, 2 * M_PI) * (r > POND_R ? 6.0 : 2.0) * dt;
        }
        /* give each other room */
        for (int j = 0; j < N_KOI; j++) {
            if (j == i)
                continue;
            double ex = k->x - kois[j].x, ey = k->y - kois[j].y, d = hypot(ex, ey);
            if (d < 60 && d > 0.1)
                k->turn += remainder(atan2(ey, ex) - k->heading, 2 * M_PI) * (60 - d) / 60 * 0.8 * dt;
        }
        k->turn = fmax(-1.6, fmin(1.6, k->turn));
        k->heading += k->turn * dt;
        k->speed += (want_speed - k->speed) * fmin(1, dt * 1.2);
        k->x += cos(k->heading) * k->speed * dt;
        k->y += sin(k->heading) * k->speed * dt;
        dx = k->x - c;
        dy = k->y - c;
        r = hypot(dx, dy);
        if (r > POND_R + 12) {                      /* never onto the rocks */
            k->x = c + dx / r * (POND_R + 12);
            k->y = c + dy / r * (POND_R + 12);
        }
        k->phase += dt * (2.2 + k->speed * 0.09);   /* tail beats faster when swimming fast */
    }

    for (int i = 0; i < MAX_RIPPLES; i++)
        if (ripples[i].alive && (ripples[i].age += dt) > 1.4)
            ripples[i].alive = 0;
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
    cairo_set_source_rgba(cr, 0.02, 0.08, 0.08, 0.8);
    cairo_stroke_preserve(cr);
    set_rgb(cr, c);
    cairo_fill(cr);
}

typedef struct { double zotac, tuf, tok; } shown_t;

/*
 * Draw a koi. The body is built upright in a small scratch surface: the sprite (head up)
 * is cut into horizontal strips, each shifted sideways by a travelling wave that grows
 * toward the tail, so the fish swims. Those copies are axis-aligned and cheap; the
 * finished fish is then rotated onto the pond in one paint, plus one for its shadow.
 */
static cairo_surface_t *koi_scratch;

static void draw_koi(cairo_t *cr, const koi *k)
{
    cairo_surface_t *img = koi_img[k->src];
    int iw = cairo_image_surface_get_width(img), ih = cairo_image_surface_get_height(img);
    int pad = iw / 2, sw = iw + 2 * pad;

    if (!koi_scratch || cairo_image_surface_get_width(koi_scratch) < sw ||
        cairo_image_surface_get_height(koi_scratch) < ih) {
        if (koi_scratch)
            cairo_surface_destroy(koi_scratch);
        koi_scratch = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, sw + 8, ih + 8);
    }
    cairo_t *sc = cairo_create(koi_scratch);
    cairo_set_operator(sc, CAIRO_OPERATOR_CLEAR);
    cairo_paint(sc);
    cairo_set_operator(sc, CAIRO_OPERATOR_OVER);
    for (int sl = 0; sl < SLICES; sl++) {
        int y0 = ih * sl / SLICES, y1 = ih * (sl + 1) / SLICES;
        double v = (y0 + y1) / 2.0 / ih;               /* 0 head .. 1 tail */
        double amp = iw * 0.16 * pow(v, 1.6) * fmin(1, 0.5 + k->speed / 60);
        int off = (int)lround(amp * sin(k->phase * 2 - v * 4.2));
        cairo_save(sc);
        cairo_rectangle(sc, 0, y0, sw, y1 - y0);
        cairo_clip(sc);
        cairo_set_source_surface(sc, img, pad + off, 0);
        cairo_paint(sc);
        cairo_restore(sc);
    }
    cairo_destroy(sc);
    cairo_surface_flush(koi_scratch);

    double scale = k->size / 1.05;
    for (int pass = 0; pass < 2; pass++) {       /* 0: soft shadow on the pond floor, 1: the fish */
        cairo_save(cr);
        cairo_translate(cr, k->x + (pass ? 0 : 7), k->y + (pass ? 0 : 10));
        cairo_rotate(cr, k->heading + M_PI / 2);  /* sprite's head points up */
        cairo_scale(cr, scale, scale);
        cairo_translate(cr, -sw / 2.0, -ih / 2.0);
        cairo_rectangle(cr, 0, 0, sw, ih);
        cairo_clip(cr);
        if (pass) {
            cairo_set_source_surface(cr, koi_scratch, 0, 0);
            cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_BILINEAR);
            cairo_paint(cr);
        } else {
            cairo_set_source_rgba(cr, 0.0, 0.05, 0.06, 0.28);
            cairo_mask_surface(cr, koi_scratch, 0, 0);
        }
        cairo_restore(cr);
    }
}

static void update_hud(const stats *s, const shown_t *sh, double t)
{
    const double c = SIZE / 2.0;
    static double tok, watts, next;
    char key[128], txt[64];
    if (t >= next || t < next - 1) {
        next = t + 0.25;
        tok = sh->tok;
        watts = s->power[0] + s->power[1];
    }
    int idle = s->tok_s < 1 && s->running == 0;
    snprintf(key, sizeof(key), "%.0f|%.0f|%d", tok, watts, idle);
    if (!strcmp(key, hud_key))
        return;
    strcpy(hud_key, key);
    cairo_t *cr = cairo_create(hud_cache);
    cairo_set_operator(cr, CAIRO_OPERATOR_CLEAR);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    if (idle)
        snprintf(txt, sizeof(txt), "still water");
    else
        snprintf(txt, sizeof(txt), "%.0f tok/s", tok);
    soft_text(cr, c, 424, 30, (rgb){ 1, 0.97, 0.90 }, txt);
    snprintf(txt, sizeof(txt), "%.0f W", watts);
    soft_text(cr, c, 456, 18, (rgb){ 0.85, 0.95, 0.92 }, txt);
    cairo_destroy(cr);
}

static void render(cairo_t *cr, const stats *s, const shown_t *sh, double t)
{
    cairo_set_source_surface(cr, pond_img, 0, 0);
    cairo_paint(cr);

    /* Ripples: rings where food lands and where the koi eat */
    cairo_set_line_width(cr, 2);
    for (int i = 0; i < MAX_RIPPLES; i++) {
        const ripple *r = &ripples[i];
        if (!r->alive)
            continue;
        double k = r->age / 1.4;
        for (int ring = 0; ring < 2; ring++) {
            double rr = r->max_r * (k - ring * 0.25);
            if (rr <= 0)
                continue;
            cairo_new_path(cr);
            cairo_arc(cr, r->x, r->y, rr, 0, 2 * M_PI);
            cairo_set_source_rgba(cr, r->col.r, r->col.g, r->col.b, 0.55 * (1 - k));
            cairo_stroke(cr);
        }
    }
    /* Food pellets floating on the surface */
    for (int i = 0; i < MAX_PELLETS; i++) {
        const pellet *p = &pellets[i];
        if (!p->alive)
            continue;
        cairo_arc(cr, p->x, p->y, 3.2, 0, 2 * M_PI);
        cairo_set_source_rgb(cr, 0.60, 0.40, 0.20);
        cairo_fill(cr);
        cairo_arc(cr, p->x - 0.8, p->y - 0.8, 1.2, 0, 2 * M_PI);
        cairo_set_source_rgba(cr, 1, 0.9, 0.7, 0.8);
        cairo_fill(cr);
    }

    for (int i = 0; i < N_KOI; i++)
        draw_koi(cr, &kois[i]);

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
    load_assets();
    init_koi();

    surf = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cr = cairo_create(surf);
    tj = tj3Init(TJINIT_COMPRESS);
    tj3Set(tj, TJPARAM_QUALITY, JPEG_QUALITY);
    tj3Set(tj, TJPARAM_SUBSAMP, TJSAMP_420);

    if (bench) {
        struct { double tok0, tok1; double w0, w1; const char *png; } scenes[] = {
            { 723, 884, 470, 460, "koi_preview.png" },
            { 180, 60, 250, 120, "koi_mid.png" },
            { 0, 0, 25, 30, "koi_idle.png" },
        };
        for (int k = 0; k < 3; k++) {
            memset(pellets, 0, sizeof(pellets));
            s.tok_port[0] = scenes[k].tok0; s.tok_port[1] = scenes[k].tok1;
            s.tok_s = s.tok_port[0] + s.tok_port[1]; s.running = s.tok_s > 0 ? 4 : 0;
            s.power[0] = scenes[k].w0; s.power[1] = scenes[k].w1;
            sh.tok = s.tok_s;
            int n = 12 * FPS_BUSY;
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
            printf("%s: %.2f ms/frame, jpeg %zu bytes\n", scenes[k].png, (now_s() - b0) * 1000 / 60, len);
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
    cairo_surface_destroy(pond_img);
    cairo_surface_destroy(koi_img[0]);
    cairo_surface_destroy(koi_img[1]);
    cairo_surface_destroy(hud_cache);
    if (nvml_ok)
        nvmlShutdown();
    if (fd >= 0)
        close(fd);
    return 0;
}
