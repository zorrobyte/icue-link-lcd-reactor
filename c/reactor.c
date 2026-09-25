/*
 * reactor: LLM dashboard for the Corsair iCUE LINK AIO pump LCD (1b1c:0c4e).
 *
 * Outer ring: GPU load (left half ZOTAC, right half TUF).
 * Core: segments that spin faster with load, radial glow that heats up with power.
 * Center: generation tokens/sec across vLLM servers, total GPU watts, GPU temps.
 *
 * Rendering: cairo -> libjpeg-turbo -> 1024-byte HID output reports on hidraw.
 * GPU stats: NVML. vLLM stats: Prometheus /metrics over plain HTTP on localhost.
 *
 * Build: see Makefile. Run with --demo to simulate data.
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
#define FPS             12
#define REPORT_SIZE     1024
#define HEADER_SIZE     8
#define CHUNK_SIZE      (REPORT_SIZE - HEADER_SIZE)
#define JPEG_QUALITY    85
#define POWER_MAX       1150.0      /* both cards near their 575 W limits */
#define N_GPUS          2
#define N_SEGMENTS      24

static const char  *gpu_bus[N_GPUS]  = { "00000000:01:00.0", "00000000:03:00.0" };  /* ZOTAC, TUF */
static const int    vllm_ports[]     = { 18090, 18091 };
#define N_PORTS     (int)(sizeof(vllm_ports) / sizeof(vllm_ports[0]))

typedef struct { double r, g, b; } rgb;

static const rgb BLUE   = { 61 / 255.0, 174 / 255.0, 233 / 255.0 };
static const rgb ORANGE = { 233 / 255.0, 120 / 255.0, 61 / 255.0 };
static const rgb WHITE  = { 240 / 255.0, 240 / 255.0, 245 / 255.0 };
static const rgb DIM    = { 110 / 255.0, 115 / 255.0, 125 / 255.0 };
static const rgb TRACK  = { 38 / 255.0, 40 / 255.0, 48 / 255.0 };
static const rgb BG     = { 6 / 255.0, 7 / 255.0, 12 / 255.0 };
static const rgb SEG_LO = { 20 / 255.0, 22 / 255.0, 30 / 255.0 };

typedef struct {
    double load[N_GPUS];        /* 0..1 */
    double power[N_GPUS];       /* W */
    int    temp[N_GPUS];        /* C */
    double tok_s;
    int    running;
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
static rgb heat_color(double t)
{
    static const rgb c0 = { 40 / 255.0, 200 / 255.0, 1.0 };
    static const rgb c1 = { 1.0, 170 / 255.0, 40 / 255.0 };
    static const rgb c2 = { 1.0, 40 / 255.0, 30 / 255.0 };
    t = clamp01(t);
    return t <= 0.55 ? lerp(c0, c1, t / 0.55) : lerp(c1, c2, (t - 0.55) / 0.45);
}

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
    static double last_tokens = -1, last_t;
    double tokens = 0;
    int running = 0, any = 0;

    if (!buf && !(buf = malloc(1 << 20)))
        return;
    for (int i = 0; i < N_PORTS; i++) {
        if (http_metrics(vllm_ports[i], buf, 1 << 20) <= 0)
            continue;
        any = 1;
        tokens  += metric_sum(buf, "vllm:generation_tokens_total");
        running += (int)metric_sum(buf, "vllm:num_requests_running");
    }
    if (!any)
        return;
    if (last_tokens >= 0 && tokens >= last_tokens && t > last_t)
        s->tok_s = 0.6 * s->tok_s + 0.4 * (tokens - last_tokens) / (t - last_t);
    last_tokens = tokens;
    last_t = t;
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
    s->tok_s    = 0.8 * s->tok_s + 0.2 * (busy > 0.15 ? 260 * busy : 0);
    s->running  = (int)(busy * 6);
}

/* ---------------------------------------------------------------- render */

static void set_rgb(cairo_t *cr, rgb c) { cairo_set_source_rgb(cr, c.r, c.g, c.b); }

/* Arc in degrees, clockwise from 3 o'clock, stroked inside radius r_outer. */
static void arc(cairo_t *cr, double r_outer, double width, double a0, double a1, rgb c)
{
    double r = r_outer - width / 2;
    cairo_set_line_width(cr, width);
    cairo_new_sub_path(cr);
    cairo_arc(cr, SIZE / 2.0, SIZE / 2.0, r, a0 * M_PI / 180, a1 * M_PI / 180);
    set_rgb(cr, c);
    cairo_stroke(cr);
}

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

static void render(cairo_t *cr, const stats *s, double phase, const shown_t *sh)
{
    const double c = SIZE / 2.0;
    double total_w = s->power[0] + s->power[1];
    double heat = clamp01(total_w / POWER_MAX);
    rgb core = heat_color(heat);
    char txt[64];

    set_rgb(cr, BG);
    cairo_paint(cr);

    /* Core glow */
    {
        double r = SIZE * (0.20 + 0.10 * heat) * 1.9;
        double a = 0.25 + 0.45 * heat;
        cairo_pattern_t *g = cairo_pattern_create_radial(c, c, 0, c, c, r);
        cairo_pattern_add_color_stop_rgba(g, 0.0, core.r, core.g, core.b, a);
        cairo_pattern_add_color_stop_rgba(g, 0.45, core.r, core.g, core.b, a * 0.6);
        cairo_pattern_add_color_stop_rgba(g, 1.0, core.r, core.g, core.b, 0);
        cairo_set_source(cr, g);
        cairo_paint(cr);
        cairo_pattern_destroy(g);
    }

    /* Outer load ring */
    {
        double ring_r = SIZE / 2.0 - 18, ring_w = 20;
        cairo_set_line_cap(cr, CAIRO_LINE_CAP_BUTT);
        arc(cr, ring_r, ring_w, 95, 265, TRACK);
        arc(cr, ring_r, ring_w, 275, 445, TRACK);
        if (sh->zotac > 0.005)
            arc(cr, ring_r, ring_w, 265 - 170 * sh->zotac, 265, BLUE);
        if (sh->tuf > 0.005)
            arc(cr, ring_r, ring_w, 275, 275 + 170 * sh->tuf, ORANGE);
    }

    /* Spinning reactor segments */
    for (int i = 0; i < N_SEGMENTS; i++) {
        double a0 = phase + i * 360.0 / N_SEGMENTS;
        double pulse = 0.35 + 0.65 * (0.5 + 0.5 * sin(a0 * 2 * M_PI / 180 + phase / 30));
        rgb col = lerp(SEG_LO, core, pulse * (0.35 + 0.65 * heat));
        arc(cr, SIZE / 2.0 - 58, 8, a0, a0 + 360.0 / N_SEGMENTS * 0.55, col);
    }

    /* Center text */
    if (s->tok_s < 0.5 && s->running == 0) {
        text_center(cr, c, c - 20, 34, 1, DIM, "IDLE");
    } else {
        snprintf(txt, sizeof(txt), "%.0f", sh->tok);
        text_center(cr, c, c - 36, 86, 1, WHITE, txt);
        text_center(cr, c, c + 18, 22, 1, core, "TOKENS / SEC");
    }
    snprintf(txt, sizeof(txt), "%.0f W", total_w);
    text_center(cr, c, c + 62, 34, 1, WHITE, txt);
    snprintf(txt, sizeof(txt), "%d\xC2\xB0", s->temp[0]);
    text_center(cr, c - 62, c + 108, 20, 0, BLUE, txt);
    snprintf(txt, sizeof(txt), "%d\xC2\xB0", s->temp[1]);
    text_center(cr, c + 62, c + 108, 20, 0, ORANGE, txt);
    if (s->running) {
        snprintf(txt, sizeof(txt), "%d req", s->running);
        text_center(cr, c, c + 108, 16, 1, DIM, txt);
    }
}

/* ---------------------------------------------------------------- main */

int main(int argc, char **argv)
{
    cairo_surface_t *surf;
    cairo_t *cr;
    tjhandle tj;
    unsigned char *jpeg = NULL;
    unsigned long jpeg_len = 0;
    stats s = { 0 };
    shown_t sh = { 0 };
    double phase = 0, last, next_poll = 0, bench_t = 0;
    int fd, bench = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--demo"))
            demo = 1;
        else if (!strcmp(argv[i], "--bench"))
            bench = 1;
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);

    surf = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cr = cairo_create(surf);
    tj = tj3Init(TJINIT_COMPRESS);
    tj3Set(tj, TJPARAM_QUALITY, JPEG_QUALITY);
    tj3Set(tj, TJPARAM_SUBSAMP, TJSAMP_420);

    if (bench) {
        /* Render + encode only, no device: report time per frame */
        s.power[0] = 440; s.power[1] = 430; s.temp[0] = 54; s.temp[1] = 71; s.tok_s = 212; s.running = 5;
        sh.zotac = 0.9; sh.tuf = 0.8; sh.tok = 212;
        double t0 = now_s();
        for (int i = 0; i < 500; i++) {
            render(cr, &s, i * 3.0, &sh);
            cairo_surface_flush(surf);
            jpeg_len = 0;
            tj3Compress8(tj, cairo_image_surface_get_data(surf), SIZE,
                         cairo_image_surface_get_stride(surf), SIZE, TJPF_BGRX, &jpeg, (size_t *)&jpeg_len);
        }
        printf("%.2f ms/frame, jpeg %lu bytes\n", (now_s() - t0) * 1000 / 500, jpeg_len);
        cairo_surface_write_to_png(surf, "reactor_c_preview.png");
        return 0;
    }

    if (!demo)
        gpus_init();

    while ((fd = lcd_open()) < 0 && !stop)
        sleep(2);
    lcd_brightness(fd, 100);

    last = now_s();
    while (!stop) {
        double t = now_s(), dt = t - last, k;
        last = t;

        if (t >= next_poll) {
            next_poll = t + 1.0;
            if (demo) {
                if (!bench_t)
                    bench_t = t;
                demo_poll(&s, t - bench_t);
            } else {
                gpus_poll(&s);
                vllm_poll(&s, t);
            }
        }

        /* Ease displayed values toward the latest readings */
        k = dt * 4 > 1 ? 1 : dt * 4;
        sh.zotac += (s.load[0] - sh.zotac) * k;
        sh.tuf   += (s.load[1] - sh.tuf) * k;
        sh.tok   += (s.tok_s - sh.tok) * k;
        phase = fmod(phase + dt * (12 + 260 * (sh.zotac + sh.tuf) / 2), 360);

        render(cr, &s, phase, &sh);
        cairo_surface_flush(surf);
        size_t len = 0;
        if (tj3Compress8(tj, cairo_image_surface_get_data(surf), SIZE, cairo_image_surface_get_stride(surf),
                         SIZE, TJPF_BGRX, &jpeg, &len) == 0) {
            if (lcd_send(fd, jpeg, len) < 0) {
                /* Device went away (sleep, replug): reopen it */
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
