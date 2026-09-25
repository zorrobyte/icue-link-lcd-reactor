/*
 * eye: a giant eyeball stares out of the iCUE LINK AIO pump LCD, watching your LLM.
 *
 * It darts around with saccades that quicken with tokens/sec and lean toward the
 * busier GPU. The pupil dilates with throughput and flinches when a prompt lands
 * after idle, snapping the lids wide open. It blinks, gets drowsy when idle and
 * falls asleep after a while. Power draw makes it bloodshot; past ~1 kW it cries.
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

/* ---------------------------------------------------------------- eye */

#define SCLERA_R        232.0
#define IRIS_R          88.0
#define CORNER_Y        250.0       /* where the lids meet at the corners */
#define N_VEINS         16
#define VEIN_PTS        10
#define N_FIBERS        140
#define IDLE_SLEEP_S    20.0        /* idle this long and the eye dozes off */

static double vein_x[N_VEINS][VEIN_PTS], vein_y[N_VEINS][VEIN_PTS], vein_w[N_VEINS];
static double fiber_a[N_FIBERS], fiber_l[N_FIBERS], fiber_b[N_FIBERS];

typedef struct {
    double gx, gy, tx, ty;          /* gaze offset of the iris (px) and saccade target */
    double next_saccade;
    double open, open_target;       /* lid openness 0 closed .. 1 open (1.15 = startled) */
    double blink_t, next_blink;     /* blink progress (0 = not blinking) */
    double pupil, startle;
    double idle_for;
    double tear_y, tear_on;
    int    was_idle;
} eye_state;

static eye_state E = { .open = 0.9, .pupil = 30, .next_blink = 2, .was_idle = 1 };

static double frand(void) { return rand() / (double)RAND_MAX; }

static void init_eye(void)
{
    const double c = SIZE / 2.0;
    for (int v = 0; v < N_VEINS; v++) {
        double a = (v + frand() * 0.6) * 2 * M_PI / N_VEINS;
        double r0 = SCLERA_R + 5, r1 = 115 + frand() * 55;
        double wob = frand() * 6;
        for (int k = 0; k < VEIN_PTS; k++) {
            double u = k / (double)(VEIN_PTS - 1);
            double r = r0 + (r1 - r0) * u;
            double aa = a + 0.12 * sin(u * 7 + wob) + (frand() - 0.5) * 0.04;
            vein_x[v][k] = c + r * cos(aa);
            vein_y[v][k] = c + r * sin(aa);
        }
        vein_w[v] = 1.0 + frand() * 1.6;
    }
    for (int i = 0; i < N_FIBERS; i++) {
        fiber_a[i] = frand() * 2 * M_PI;
        fiber_l[i] = 0.55 + frand() * 0.45;
        fiber_b[i] = frand();
    }
}

static void simulate(const stats *s, double dt, double t)
{
    int idle = s->tok_s < 1 && s->running == 0;
    double tok = s->tok_s;

    E.idle_for = idle ? E.idle_for + dt : 0;

    /* A prompt arriving after idle: startle */
    if (E.was_idle && !idle)
        E.startle = 1;
    E.was_idle = idle;
    E.startle *= exp(-dt * 1.8);

    /* Lids: drowsy when idle, asleep after a while, wide when busy, wider when startled */
    if (idle)
        E.open_target = E.idle_for > IDLE_SLEEP_S ? 0 : 0.42 - 0.2 * clamp01(E.idle_for / IDLE_SLEEP_S);
    else
        E.open_target = 0.88 + 0.07 * clamp01(tok / 1500) + 0.3 * E.startle;
    E.open += (E.open_target - E.open) * fmin(1, dt * (idle ? 1.2 : 6));

    /* Blinks */
    if (t >= E.next_blink && E.open > 0.2) {
        E.blink_t = 0.001;
        E.next_blink = t + 2.5 + frand() * 4;
    }
    if (E.blink_t > 0) {
        E.blink_t += dt;
        if (E.blink_t > 0.22)
            E.blink_t = 0;
    }

    /* Saccades: more often with throughput, leaning toward the busier GPU */
    if (t >= E.next_saccade) {
        double bias = (s->tok_port[1] - s->tok_port[0]) / (s->tok_port[0] + s->tok_port[1] + 1) * 55;
        double a = frand() * 2 * M_PI, r = sqrt(frand());
        E.tx = bias + cos(a) * r * 80;
        E.ty = sin(a) * r * 45 - 5;
        E.next_saccade = t + (idle ? 2.5 + frand() * 3 : fmax(0.25, 1.6 - tok / 1200) * (0.5 + frand()));
    }
    E.gx += (E.tx - E.gx) * fmin(1, dt * 16);
    E.gy += (E.ty - E.gy) * fmin(1, dt * 16);

    /* Pupil dilates with throughput, flinches small when startled */
    double want = 24 + 22 * clamp01(tok / 1600) - 12 * E.startle;
    E.pupil += (want - E.pupil) * fmin(1, dt * 3);

    /* Tears when it's working very hard */
    double watts = s->power[0] + s->power[1];
    if (E.tear_on <= 0 && watts > 1000 && frand() < dt * 0.25) {
        E.tear_on = 1;
        E.tear_y = 0;
    }
    if (E.tear_on > 0) {
        E.tear_y += dt * (30 + E.tear_y * 0.8);
        if (E.tear_y > 200)
            E.tear_on = 0;
    }
}

/* ---------------------------------------------------------------- render */

static void set_rgb(cairo_t *cr, rgb c) { cairo_set_source_rgb(cr, c.r, c.g, c.b); }

static void text_center(cairo_t *cr, double x, double y, double size, int bold, rgb c, const char *s)
{
    cairo_text_extents_t ext;
    cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL,
                           bold ? CAIRO_FONT_WEIGHT_BOLD : CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, size);
    cairo_text_extents(cr, s, &ext);
    double tx = x - ext.width / 2 - ext.x_bearing, ty = y - ext.height / 2 - ext.y_bearing;
    cairo_move_to(cr, tx + 1.5, ty + 1.5);
    cairo_set_source_rgba(cr, 0.25, 0.10, 0.08, 0.55);
    cairo_show_text(cr, s);
    cairo_move_to(cr, tx, ty);
    set_rgb(cr, c);
    cairo_show_text(cr, s);
}

typedef struct { double zotac, tuf, tok; } shown_t;

/* Lid edge curves for the current openness: upper lid y at x, lower lid y at x */
static void lid_curves(double open, double *up_c, double *lo_c)
{
    *up_c = CORNER_Y - 262 * open;
    *lo_c = CORNER_Y + 170 * fmin(open, 1.0) + 25 * fmax(0, open - 1);
}

static void upper_lid_path(cairo_t *cr, double up_c)
{
    cairo_move_to(cr, -20, CORNER_Y + 6);
    cairo_line_to(cr, 34, CORNER_Y);
    cairo_curve_to(cr, 150, up_c, 330, up_c, 446, CORNER_Y);
    cairo_line_to(cr, SIZE + 20, CORNER_Y + 6);
}

static void lower_lid_path(cairo_t *cr, double lo_c)
{
    cairo_move_to(cr, -20, CORNER_Y + 6);
    cairo_line_to(cr, 34, CORNER_Y);
    cairo_curve_to(cr, 150, lo_c, 330, lo_c, 446, CORNER_Y);
    cairo_line_to(cr, SIZE + 20, CORNER_Y + 6);
}

static void render(cairo_t *cr, const stats *s, const shown_t *sh, double t)
{
    const double c = SIZE / 2.0;
    double total_w = s->power[0] + s->power[1];
    int tmax = s->temp[0] > s->temp[1] ? s->temp[0] : s->temp[1];
    double bloodshot = clamp01((total_w - 250) / 750) * 0.85 + clamp01((tmax - 60) / 25.0) * 0.15;
    const rgb SKIN = { 0.91, 0.70, 0.60 }, SKIN_DARK = { 0.72, 0.48, 0.40 };
    char txt[64];

    /* Blink closes the lids briefly on top of the current openness */
    double open = E.open;
    if (E.blink_t > 0) {
        double b = E.blink_t / 0.22;
        open *= fabs(1 - 2 * b) * 0.95 + 0.05 * (b > 0.5);
    }
    double up_c, lo_c;
    lid_curves(open, &up_c, &lo_c);

    /* Eyeball: sclera with soft shading */
    {
        cairo_pattern_t *g = cairo_pattern_create_radial(c - 30, c - 40, 30, c, c, SCLERA_R + 10);
        cairo_pattern_add_color_stop_rgb(g, 0.0, 1.0, 0.99, 0.97);
        cairo_pattern_add_color_stop_rgb(g, 0.7, 0.93, 0.90, 0.88);
        cairo_pattern_add_color_stop_rgb(g, 1.0, 0.80, 0.70, 0.68);
        cairo_set_source(cr, g);
        cairo_paint(cr);
        cairo_pattern_destroy(g);
    }

    /* Veins, more and redder with power draw */
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
    for (int v = 0; v < N_VEINS; v++) {
        double a = 0.10 + 0.75 * bloodshot * (0.5 + 0.5 * ((v * 7) % 5) / 4.0);
        cairo_new_path(cr);
        for (int k = 0; k < VEIN_PTS; k++)
            cairo_line_to(cr, vein_x[v][k], vein_y[v][k]);
        cairo_set_line_width(cr, vein_w[v] * (0.7 + 0.8 * bloodshot));
        cairo_set_source_rgba(cr, 0.80, 0.12, 0.12, a);
        cairo_stroke(cr);
        /* a small branch */
        int k0 = VEIN_PTS / 3;
        cairo_move_to(cr, vein_x[v][k0], vein_y[v][k0]);
        cairo_line_to(cr, vein_x[v][k0] + (vein_y[v][k0 + 2] - vein_y[v][k0]) * 0.8,
                      vein_y[v][k0] - (vein_x[v][k0 + 2] - vein_x[v][k0]) * 0.8);
        cairo_set_line_width(cr, vein_w[v] * 0.5 * (0.7 + 0.8 * bloodshot));
        cairo_stroke(cr);
    }

    /* Iris, foreshortened as it turns away from center */
    {
        double ix = c + E.gx, iy = c + E.gy;
        double sx = 1 - fabs(E.gx) / 420, sy = 1 - fabs(E.gy) / 420;
        cairo_save(cr);
        cairo_translate(cr, ix, iy);
        cairo_scale(cr, sx, sy);

        cairo_pattern_t *g = cairo_pattern_create_radial(0, 0, E.pupil * 0.8, 0, 0, IRIS_R);
        cairo_pattern_add_color_stop_rgb(g, 0.00, 0.85, 0.62, 0.22);   /* amber around the pupil */
        cairo_pattern_add_color_stop_rgb(g, 0.35, 0.40, 0.55, 0.30);
        cairo_pattern_add_color_stop_rgb(g, 0.75, 0.12, 0.45, 0.50);   /* teal */
        cairo_pattern_add_color_stop_rgb(g, 0.93, 0.06, 0.22, 0.26);
        cairo_pattern_add_color_stop_rgb(g, 1.00, 0.02, 0.06, 0.08);   /* limbal ring */
        cairo_arc(cr, 0, 0, IRIS_R, 0, 2 * M_PI);
        cairo_set_source(cr, g);
        cairo_fill(cr);
        cairo_pattern_destroy(g);

        /* Fibers */
        cairo_set_line_width(cr, 1.3);
        for (int i = 0; i < N_FIBERS; i++) {
            double a = fiber_a[i], r0 = E.pupil + 2, r1 = r0 + (IRIS_R - 6 - r0) * fiber_l[i];
            cairo_move_to(cr, r0 * cos(a), r0 * sin(a));
            cairo_line_to(cr, r1 * cos(a + 0.05), r1 * sin(a + 0.05));
            if (fiber_b[i] > 0.5)
                cairo_set_source_rgba(cr, 1.0, 0.92, 0.70, 0.22);
            else
                cairo_set_source_rgba(cr, 0.0, 0.08, 0.06, 0.30);
            cairo_stroke(cr);
        }

        /* Pupil */
        cairo_arc(cr, 0, 0, E.pupil, 0, 2 * M_PI);
        cairo_set_source_rgb(cr, 0.01, 0.01, 0.02);
        cairo_fill(cr);

        /* Catchlights */
        cairo_save(cr);
        cairo_translate(cr, -IRIS_R * 0.33, -IRIS_R * 0.38);
        cairo_scale(cr, 1.3, 1);
        cairo_arc(cr, 0, 0, 13, 0, 2 * M_PI);
        cairo_restore(cr);
        cairo_set_source_rgba(cr, 1, 1, 1, 0.9);
        cairo_fill(cr);
        cairo_arc(cr, IRIS_R * 0.32, IRIS_R * 0.30, 5, 0, 2 * M_PI);
        cairo_set_source_rgba(cr, 1, 1, 1, 0.55);
        cairo_fill(cr);
        cairo_restore(cr);
    }

    /* Shadow the lids cast on the eyeball */
    cairo_save(cr);
    cairo_new_path(cr);
    upper_lid_path(cr, up_c + 22);
    cairo_set_line_width(cr, 46);
    cairo_set_source_rgba(cr, 0.35, 0.18, 0.15, 0.22);
    cairo_stroke(cr);
    cairo_new_path(cr);
    lower_lid_path(cr, lo_c - 10);
    cairo_set_line_width(cr, 20);
    cairo_set_source_rgba(cr, 0.35, 0.18, 0.15, 0.15);
    cairo_stroke(cr);
    cairo_restore(cr);

    /* Upper lid (skin above the curve) */
    {
        cairo_new_path(cr);
        upper_lid_path(cr, up_c);
        cairo_line_to(cr, SIZE + 20, -20);
        cairo_line_to(cr, -20, -20);
        cairo_close_path(cr);
        cairo_pattern_t *g = cairo_pattern_create_linear(0, up_c - 60, 0, CORNER_Y);
        cairo_pattern_add_color_stop_rgb(g, 0, SKIN.r, SKIN.g, SKIN.b);
        cairo_pattern_add_color_stop_rgb(g, 1, SKIN_DARK.r, SKIN_DARK.g, SKIN_DARK.b);
        cairo_set_source(cr, g);
        cairo_fill(cr);
        cairo_pattern_destroy(g);
        /* crease above the lid */
        cairo_new_path(cr);
        cairo_move_to(cr, 60, CORNER_Y - 20);
        cairo_curve_to(cr, 150, up_c - 55, 330, up_c - 55, 420, CORNER_Y - 20);
        cairo_set_line_width(cr, 3);
        cairo_set_source_rgba(cr, 0.55, 0.32, 0.26, 0.45);
        cairo_stroke(cr);
    }

    /* Lower lid (skin below the curve) */
    {
        cairo_new_path(cr);
        lower_lid_path(cr, lo_c);
        cairo_line_to(cr, SIZE + 20, SIZE + 20);
        cairo_line_to(cr, -20, SIZE + 20);
        cairo_close_path(cr);
        cairo_pattern_t *g = cairo_pattern_create_linear(0, CORNER_Y, 0, SIZE);
        cairo_pattern_add_color_stop_rgb(g, 0, SKIN_DARK.r, SKIN_DARK.g, SKIN_DARK.b);
        cairo_pattern_add_color_stop_rgb(g, 0.4, SKIN.r, SKIN.g, SKIN.b);
        cairo_pattern_add_color_stop_rgb(g, 1, SKIN.r * 0.95, SKIN.g * 0.93, SKIN.b * 0.93);
        cairo_set_source(cr, g);
        cairo_fill(cr);
        cairo_pattern_destroy(g);
    }

    /* Lid rims */
    cairo_new_path(cr);
    upper_lid_path(cr, up_c);
    cairo_set_line_width(cr, 4);
    cairo_set_source_rgb(cr, 0.35, 0.16, 0.13);
    cairo_stroke(cr);
    cairo_new_path(cr);
    lower_lid_path(cr, lo_c);
    cairo_set_line_width(cr, 2.5);
    cairo_set_source_rgba(cr, 0.60, 0.30, 0.26, 0.8);
    cairo_stroke(cr);

    /* Lashes along the upper lid, pointing down when shut */
    {
        double lash_dir = open < 0.08 ? 1 : -1;
        cairo_set_line_width(cr, 2.2);
        cairo_set_source_rgb(cr, 0.10, 0.05, 0.04);
        for (int i = 1; i < 24; i++) {
            double u = i / 24.0;
            /* Point on the bezier from (34,CORNER_Y) to (446,CORNER_Y) with control y = up_c */
            double mu = 1 - u;
            double x = mu * mu * mu * 34 + 3 * mu * mu * u * 150 + 3 * mu * u * u * 330 + u * u * u * 446;
            double y = mu * mu * mu * CORNER_Y + 3 * mu * mu * u * up_c + 3 * mu * u * u * up_c + u * u * u * CORNER_Y;
            double len = 12 + 10 * sin(M_PI * u);
            double lean = (u - 0.5) * 16;
            cairo_move_to(cr, x, y);
            cairo_curve_to(cr, x + lean * 0.3, y + lash_dir * len * 0.6, x + lean, y + lash_dir * len * 0.9,
                           x + lean * 1.4, y + lash_dir * len);
            cairo_stroke(cr);
        }
    }

    /* Tear rolling from the inner corner */
    if (E.tear_on > 0) {
        double tx = 44 + E.tear_y * 0.15, ty = CORNER_Y + 10 + E.tear_y;
        cairo_move_to(cr, tx, ty - 9);
        cairo_curve_to(cr, tx + 6, ty - 1, tx + 6, ty + 6, tx, ty + 6);
        cairo_curve_to(cr, tx - 6, ty + 6, tx - 6, ty - 1, tx, ty - 9);
        cairo_set_source_rgba(cr, 0.75, 0.90, 1.0, 0.75);
        cairo_fill(cr);
        cairo_arc(cr, tx - 1.5, ty + 1, 1.6, 0, 2 * M_PI);
        cairo_set_source_rgba(cr, 1, 1, 1, 0.9);
        cairo_fill(cr);
    }

    /* Sleeping */
    if (open < 0.05) {
        cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
        for (int i = 0; i < 3; i++) {
            double ph = fmod(t * 0.5 + i / 3.0, 1.0);
            cairo_set_font_size(cr, 16 + ph * 22);
            cairo_move_to(cr, 300 + ph * 60, 190 - ph * 110);
            cairo_set_source_rgba(cr, 0.35, 0.16, 0.13, 1 - ph);
            cairo_show_text(cr, "Z");
        }
    }

    /* Stats on the cheek */
    if (s->tok_s < 0.5 && s->running == 0)
        snprintf(txt, sizeof(txt), "idle");
    else
        snprintf(txt, sizeof(txt), "%.0f tok/s", sh->tok);
    text_center(cr, c, 424, 24, 1, (rgb){ 1, 0.97, 0.94 }, txt);
    snprintf(txt, sizeof(txt), "%.0f W", total_w);
    text_center(cr, c, 452, 17, 1, (rgb){ 1, 0.95, 0.9 }, txt);
    snprintf(txt, sizeof(txt), "%d\xC2\xB0", s->temp[0]);
    text_center(cr, c - 58, 450, 17, 1, (rgb){ 0.20, 0.45, 0.95 }, txt);
    snprintf(txt, sizeof(txt), "%d\xC2\xB0", s->temp[1]);
    text_center(cr, c + 58, 450, 17, 1, (rgb){ 0.95, 0.45, 0.10 }, txt);
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
    init_eye();

    surf = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cr = cairo_create(surf);
    tj = tj3Init(TJINIT_COMPRESS);
    tj3Set(tj, TJPARAM_QUALITY, JPEG_QUALITY);
    tj3Set(tj, TJPARAM_SUBSAMP, TJSAMP_420);

    if (bench) {
        struct { double tok0, tok1; double w0, w1; double settle; const char *png; } scenes[] = {
            { 723, 884, 520, 510, 6, "eye_preview.png" },       /* hard at work, bloodshot */
            { 150, 40, 200, 90, 6, "eye_mid.png" },
            { 0, 0, 25, 30, 8, "eye_idle.png" },                /* drowsy */
            { 0, 0, 25, 30, 30, "eye_sleep.png" },              /* asleep */
        };
        for (int k = 0; k < 4; k++) {
            memset(&E, 0, sizeof(E));
            E.open = 0.9; E.pupil = 30; E.next_blink = 1e9; E.was_idle = scenes[k].tok0 + scenes[k].tok1 < 1;
            s.tok_port[0] = scenes[k].tok0; s.tok_port[1] = scenes[k].tok1;
            s.tok_s = s.tok_port[0] + s.tok_port[1]; s.running = s.tok_s > 0 ? 4 : 0;
            s.power[0] = scenes[k].w0; s.power[1] = scenes[k].w1; s.temp[0] = 54; s.temp[1] = 71;
            sh.tok = s.tok_s;
            int n = (int)(scenes[k].settle * FPS);
            size_t len = 0;
            double b0 = 0;
            for (int i = 0; i < n; i++) {
                if (i == n - 60)
                    b0 = now_s();
                simulate(&s, 1.0 / FPS, i / (double)FPS);
                render(cr, &s, &sh, i / (double)FPS);
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
                demo_poll(&s, t - t0);
                s.tok_port[0] *= 6; s.tok_port[1] *= 6;
                s.tok_s = s.tok_port[0] + s.tok_port[1];
                if (s.tok_s < 150) { s.tok_s = s.tok_port[0] = s.tok_port[1] = 0; s.running = 0; }
                s.power[0] *= 1.2; s.power[1] *= 1.2;
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

        double spare = 1.0 / FPS - (now_s() - t);
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
