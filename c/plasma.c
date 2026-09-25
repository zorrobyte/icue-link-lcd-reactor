/*
 * plasma: your LLM servers as a plasma globe on the iCUE LINK AIO pump LCD.
 *
 * One filament per request slot (vLLM --max-num-seqs) on each server: violet on the
 * left for the ZOTAC's, pink on the right for the TUF's. Busy slots crackle brightly
 * and carry token pulses at the server's real rate; free slots drift faint and dim,
 * so the globe shows both what's running and the spare capacity.
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
static const rgb WHITE  = { 240 / 255.0, 240 / 255.0, 245 / 255.0 };
static const rgb DIM    = { 110 / 255.0, 115 / 255.0, 125 / 255.0 };
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

/* ---------------------------------------------------------------- plasma */

#define ELECTRODE_R     62.0        /* center electrode radius (px) */
#define GLASS_R         232.0       /* where filaments hit the glass */
#define FIL_PTS         48
#define MAX_PULSES      400
#define TOKENS_PER_PULSE 12.0

#define SLOTS_PER_SERVER 4          /* vLLM --max-num-seqs on each server */
#define N_SLOTS         (SLOTS_PER_SERVER * 2)
#define FREE_GLOW       0.22        /* brightness of an idle slot */

typedef struct {
    double theta, home;     /* current / resting angle where it meets the glass */
    double phase[3];        /* noise phases for its shape */
    double life;            /* brightness, eased toward 1 when busy, FREE_GLOW when free */
    double jit;             /* this filament's crackle, eased */
    int    src;             /* 0 ZOTAC, 1 TUF */
    int    busy;
    double px[FIL_PTS], py[FIL_PTS];
} filament;

typedef struct { int fil; double u; int alive; } pulse;

static filament fils[N_SLOTS];
static pulse    pulses[MAX_PULSES];
static double   pulse_acc[2];

static double frand(void) { return rand() / (double)RAND_MAX; }

static rgb fil_color(int src)
{
    static const rgb VIOLET = { 0.55, 0.45, 1.0 };      /* ZOTAC */
    static const rgb MAGENTA = { 1.0, 0.45, 0.72 };     /* TUF */
    return src == 0 ? VIOLET : MAGENTA;
}

/* Slots fan evenly across each server's half of the globe */
static void init_slots(void)
{
    for (int i = 0; i < N_SLOTS; i++) {
        filament *f = &fils[i];
        int src = i / SLOTS_PER_SERVER, k = i % SLOTS_PER_SERVER;
        double spread = 2.2, off = (k - (SLOTS_PER_SERVER - 1) / 2.0) * spread / (SLOTS_PER_SERVER - 1);
        const double lift = 0.28;                           /* keep the lowest slots clear of the stats */
        f->src = src;
        f->home = f->theta = (src == 0 ? M_PI - off + lift : off - lift);   /* slot 1 at the top on both sides */
        for (int j = 0; j < 3; j++)
            f->phase[j] = frand() * 100;
        f->life = FREE_GLOW;
        f->jit = 0.2;
    }
}

static void spawn_pulse(int src)
{
    int cand[N_SLOTS], n = 0;
    for (int i = 0; i < N_SLOTS; i++)
        if (fils[i].src == src && fils[i].busy)
            cand[n++] = i;
    if (!n)
        for (int i = 0; i < N_SLOTS; i++)
            if (fils[i].src == src)
                cand[n++] = i;
    for (int i = 0; i < MAX_PULSES; i++)
        if (!pulses[i].alive) {
            pulses[i] = (pulse){ cand[rand() % n], 0, 1 };
            return;
        }
}

static void simulate(const stats *s, double dt, double t, double *jitter)
{
    /* How violently busy filaments crackle */
    *jitter = 0.6 + fmin(s->tok_s, 2400) / 1200;

    for (int i = 0; i < N_SLOTS; i++) {
        filament *f = &fils[i];
        int k = i % SLOTS_PER_SERVER;
        f->busy = k < s->running_port[f->src];
        double want = f->busy ? 1 : FREE_GLOW, want_jit = f->busy ? *jitter : 0.2;
        f->life += (want - f->life) * fmin(1, dt * 4);
        f->jit += (want_jit - f->jit) * fmin(1, dt * 3);

        /* Wander a little around its home angle */
        f->theta = f->home + 0.12 * sin(t * (f->busy ? 0.7 : 0.25) + f->phase[0]);

        for (int j = 0; j < FIL_PTS; j++) {
            double u = j / (double)(FIL_PTS - 1);
            double r = ELECTRODE_R + (GLASS_R - ELECTRODE_R) * u;
            double pin = sin(M_PI * u);
            double speed = f->busy ? 1 : 0.35;
            double wander = 0.24 * sin(u * 4.3 + t * 0.9 * speed + f->phase[0]) + 0.09 * sin(u * 9.7 - t * 1.7 * speed + f->phase[1]);
            double crackle = (frand() - 0.5) * 0.028 * f->jit + 0.022 * f->jit * sin(u * 31 + t * 23 + f->phase[2]);
            double a = f->theta + (wander + crackle) * pin * (0.55 + 0.45 * u);
            f->px[j] = SIZE / 2.0 + r * cos(a);
            f->py[j] = SIZE / 2.0 + r * sin(a);
        }
    }

    for (int src = 0; src < 2; src++) {
        pulse_acc[src] += s->tok_port[src] / TOKENS_PER_PULSE * dt;
        while (pulse_acc[src] >= 1) {
            spawn_pulse(src);
            pulse_acc[src] -= 1;
        }
    }
    for (int i = 0; i < MAX_PULSES; i++) {
        pulse *p = &pulses[i];
        if (!p->alive)
            continue;
        p->u += dt * 1.6;
        if (p->u >= 1)
            p->alive = 0;
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
    cairo_move_to(cr, x - ext.width / 2 - ext.x_bearing, y - ext.height / 2 - ext.y_bearing);
    set_rgb(cr, c);
    cairo_show_text(cr, s);
}

typedef struct { double zotac, tuf, tok; } shown_t;

static void fil_path(cairo_t *cr, const filament *f, int from, int to)
{
    cairo_new_path(cr);
    cairo_move_to(cr, f->px[from], f->py[from]);
    for (int k = from + 1; k <= to; k++)
        cairo_line_to(cr, f->px[k], f->py[k]);
}

static void render(cairo_t *cr, const stats *s, const shown_t *sh, double t, double jitter)
{
    const double c = SIZE / 2.0;
    double total_w = s->power[0] + s->power[1];
    char txt[64];

    /* Glass sphere interior: deep purple, a little brighter toward the middle */
    {
        cairo_pattern_t *g = cairo_pattern_create_radial(c, c, 20, c, c, GLASS_R + 10);
        cairo_pattern_add_color_stop_rgb(g, 0.0, 0.10, 0.04, 0.18);
        cairo_pattern_add_color_stop_rgb(g, 0.7, 0.04, 0.02, 0.09);
        cairo_pattern_add_color_stop_rgb(g, 1.0, 0.02, 0.01, 0.05);
        cairo_set_source(cr, g);
        cairo_paint(cr);
        cairo_pattern_destroy(g);
    }

    /* Filaments: soft glow, colored body, white-hot core; flicker with the crackle */
    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
    for (int i = 0; i < N_SLOTS; i++) {
        const filament *f = &fils[i];
        rgb col = fil_color(f->src);
        double flicker = f->busy ? 0.8 + 0.2 * sin(t * (14 + 10 * jitter) + i * 1.7) : 1;
        double a = f->life * flicker;

        fil_path(cr, f, 0, FIL_PTS - 1);
        cairo_set_line_width(cr, 14);
        cairo_set_source_rgba(cr, col.r, col.g, col.b, 0.08 * a);
        cairo_stroke_preserve(cr);
        cairo_set_line_width(cr, 5);
        cairo_set_source_rgba(cr, col.r, col.g, col.b, 0.35 * a);
        cairo_stroke_preserve(cr);
        cairo_set_line_width(cr, 1.6);
        cairo_set_source_rgba(cr, 1, 0.95, 1, 0.75 * a);
        cairo_stroke(cr);

        /* A short fork near the glass now and then, on busy slots */
        if (f->busy && ((int)(t * 9) + i) % 3 == 0) {
            int k0 = FIL_PTS * 2 / 3;
            double dx = f->px[FIL_PTS - 1] - f->px[k0], dy = f->py[FIL_PTS - 1] - f->py[k0];
            double sgn = (i & 1) ? 1 : -1;
            cairo_new_path(cr);
            cairo_move_to(cr, f->px[k0], f->py[k0]);
            cairo_line_to(cr, f->px[k0] + dx * 0.55 - sgn * dy * 0.35, f->py[k0] + dy * 0.55 + sgn * dx * 0.35);
            cairo_set_line_width(cr, 3);
            cairo_set_source_rgba(cr, col.r, col.g, col.b, 0.35 * a);
            cairo_stroke_preserve(cr);
            cairo_set_line_width(cr, 1);
            cairo_set_source_rgba(cr, 1, 0.95, 1, 0.5 * a);
            cairo_stroke(cr);
        }

        /* Hot spot where it touches the glass */
        double ex = f->px[FIL_PTS - 1], ey = f->py[FIL_PTS - 1];
        double hot = f->busy ? 26 : 12;
        cairo_pattern_t *g = cairo_pattern_create_radial(ex, ey, 0, ex, ey, hot);
        cairo_pattern_add_color_stop_rgba(g, 0, 1, 0.9, 1, 0.55 * a);
        cairo_pattern_add_color_stop_rgba(g, 0.3, col.r, col.g, col.b, 0.35 * a);
        cairo_pattern_add_color_stop_rgba(g, 1, col.r, col.g, col.b, 0);
        cairo_set_source(cr, g);
        cairo_arc(cr, ex, ey, hot, 0, 2 * M_PI);
        cairo_fill(cr);
        cairo_pattern_destroy(g);
    }

    /* Token pulses */
    for (int i = 0; i < MAX_PULSES; i++) {
        const pulse *p = &pulses[i];
        if (!p->alive)
            continue;
        const filament *f = &fils[p->fil];
        double fk = p->u * (FIL_PTS - 1);
        int k = (int)fk;
        if (k >= FIL_PTS - 1)
            continue;
        (void)0;
        double fr = fk - k;
        double x = f->px[k] + (f->px[k + 1] - f->px[k]) * fr, y = f->py[k] + (f->py[k + 1] - f->py[k]) * fr;
        rgb col = fil_color(f->src);
        cairo_pattern_t *g = cairo_pattern_create_radial(x, y, 0, x, y, 6);
        cairo_pattern_add_color_stop_rgba(g, 0, 1, 1, 1, 0.9 * fmax(f->life, 0.6));
        cairo_pattern_add_color_stop_rgba(g, 1, col.r, col.g, col.b, 0);
        cairo_set_source(cr, g);
        cairo_arc(cr, x, y, 6, 0, 2 * M_PI);
        cairo_fill(cr);
        cairo_pattern_destroy(g);
    }

    /* Electrode glow */
    {
        double a = s->tok_s < 1 && s->running == 0 ? 0.35 : 0.55 + 0.15 * sin(t * 7);
        cairo_pattern_t *g = cairo_pattern_create_radial(c, c, ELECTRODE_R * 0.8, c, c, ELECTRODE_R * 2.1);
        cairo_pattern_add_color_stop_rgba(g, 0, 0.85, 0.6, 1.0, a);
        cairo_pattern_add_color_stop_rgba(g, 1, 0.55, 0.3, 1.0, 0);
        cairo_set_source(cr, g);
        cairo_paint(cr);
        cairo_pattern_destroy(g);
    }
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);

    /* Electrode ball: dark glass with a rim light */
    {
        cairo_pattern_t *g = cairo_pattern_create_radial(c - 18, c - 22, 6, c, c, ELECTRODE_R);
        cairo_pattern_add_color_stop_rgb(g, 0, 0.30, 0.22, 0.40);
        cairo_pattern_add_color_stop_rgb(g, 0.6, 0.08, 0.05, 0.13);
        cairo_pattern_add_color_stop_rgb(g, 1, 0.03, 0.02, 0.06);
        cairo_arc(cr, c, c, ELECTRODE_R, 0, 2 * M_PI);
        cairo_set_source(cr, g);
        cairo_fill(cr);
        cairo_pattern_destroy(g);
        cairo_arc(cr, c, c, ELECTRODE_R, 0, 2 * M_PI);
        cairo_set_line_width(cr, 2);
        cairo_set_source_rgba(cr, 0.9, 0.75, 1, 0.8);
        cairo_stroke(cr);
    }

    if (s->tok_s < 0.5 && s->running == 0) {
        text_center(cr, c, c, 22, 1, DIM, "IDLE");
    } else {
        snprintf(txt, sizeof(txt), "%.0f", sh->tok);
        text_center(cr, c, c - 8, sh->tok >= 1000 ? 36 : 46, 1, WHITE, txt);
        text_center(cr, c, c + 24, 14, 1, (rgb){ 0.85, 0.7, 1.0 }, "TOK/S");
    }

    /* Glass: rim and a specular highlight */
    cairo_new_path(cr);
    cairo_arc(cr, c, c, GLASS_R + 3, 0, 2 * M_PI);
    cairo_set_line_width(cr, 3);
    cairo_set_source_rgba(cr, 0.75, 0.65, 1.0, 0.35);
    cairo_stroke(cr);
    cairo_new_path(cr);
    cairo_arc(cr, c, c, GLASS_R - 14, M_PI * 1.08, M_PI * 1.42);
    cairo_set_line_width(cr, 7);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    cairo_set_source_rgba(cr, 1, 1, 1, 0.16);
    cairo_stroke(cr);

    /* Bottom stats over a soft shade */
    {
        cairo_pattern_t *g = cairo_pattern_create_linear(0, c + 105, 0, c + 190);
        cairo_pattern_add_color_stop_rgba(g, 0.0, 0.02, 0.01, 0.05, 0);
        cairo_pattern_add_color_stop_rgba(g, 0.5, 0.02, 0.01, 0.05, 0.75);
        cairo_pattern_add_color_stop_rgba(g, 1.0, 0.02, 0.01, 0.05, 0.85);
        cairo_rectangle(cr, 0, c + 105, SIZE, SIZE);
        cairo_set_source(cr, g);
        cairo_fill(cr);
        cairo_pattern_destroy(g);
    }
    snprintf(txt, sizeof(txt), "%.0f W", total_w);
    text_center(cr, c, c + 138, 34, 1, WHITE, txt);
    snprintf(txt, sizeof(txt), "%d\xC2\xB0", s->temp[0]);
    text_center(cr, c - 72, c + 180, 26, 1, fil_color(0), txt);
    snprintf(txt, sizeof(txt), "%d\xC2\xB0", s->temp[1]);
    text_center(cr, c + 72, c + 180, 26, 1, fil_color(1), txt);
    snprintf(txt, sizeof(txt), "%d/%d  %d/%d", s->running_port[0], SLOTS_PER_SERVER, s->running_port[1], SLOTS_PER_SERVER);
    text_center(cr, c, c + 180, 16, 1, DIM, txt);
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
    double last, next_poll = 0, t0, jitter = 0;
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

    init_slots();
    surf = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cr = cairo_create(surf);
    tj = tj3Init(TJINIT_COMPRESS);
    tj3Set(tj, TJPARAM_QUALITY, JPEG_QUALITY);
    tj3Set(tj, TJPARAM_SUBSAMP, TJSAMP_420);

    if (bench) {
        struct { double tok0, tok1; int run0, run1; double w0, w1; const char *png; } scenes[] = {
            { 723, 884, 4, 4, 470, 460, "plasma_preview.png" },
            { 420, 190, 3, 1, 330, 250, "plasma_one.png" },
            { 0, 0, 0, 0, 25, 30, "plasma_idle.png" },
        };
        for (int k = 0; k < 3; k++) {
            memset(pulses, 0, sizeof(pulses));
            s.tok_port[0] = scenes[k].tok0; s.tok_port[1] = scenes[k].tok1;
            s.running_port[0] = scenes[k].run0; s.running_port[1] = scenes[k].run1;
            s.tok_s = s.tok_port[0] + s.tok_port[1]; s.running = scenes[k].run0 + scenes[k].run1;
            s.power[0] = scenes[k].w0; s.power[1] = scenes[k].w1; s.temp[0] = 54; s.temp[1] = 71;
            sh.tok = s.tok_s;
            double b0 = now_s();
            int n = 90;
            size_t len = 0;
            for (int i = 0; i < n; i++) {
                simulate(&s, 1.0 / FPS, i / (double)FPS, &jitter);
                render(cr, &s, &sh, i / (double)FPS, jitter);
                cairo_surface_flush(surf);
                tj3Compress8(tj, cairo_image_surface_get_data(surf), SIZE, cairo_image_surface_get_stride(surf),
                             SIZE, TJPF_BGRX, &jpeg, &len);
            }
            printf("%s: %.2f ms/frame, jpeg %zu bytes\n", scenes[k].png, (now_s() - b0) * 1000 / n, len);
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
                if (s.tok_s < 60) s.tok_s = s.tok_port[0] = s.tok_port[1] = 0;
                for (int j = 0; j < 2; j++)
                    s.running_port[j] = s.tok_port[j] < 30 ? 0 : 1 + (int)fmin(SLOTS_PER_SERVER - 1, s.tok_port[j] / 220);
                s.running = s.running_port[0] + s.running_port[1];
            } else {
                gpus_poll(&s);
                vllm_poll(&s, t);
            }
        }

        k = dt * 4 > 1 ? 1 : dt * 4;
        sh.tok += (s.tok_s - sh.tok) * k;

        simulate(&s, dt, t - t0, &jitter);
        render(cr, &s, &sh, t - t0, jitter);
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
