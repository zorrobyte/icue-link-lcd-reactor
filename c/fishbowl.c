/*
 * fishbowl: an aquarium of pet fish on the iCUE LINK AIO pump LCD, fed by your LLM.
 *
 * Six pet fish live in the tank. Generated tokens from every vLLM server sprinkle in
 * as food flakes at random spots on the surface (one per 25 tokens); the fish perk up,
 * chase and eat them. Light rays, caustics, swaying seaweed, a bubbler and a very slow
 * snail. Idle, the fish cruise lazily. Watts and GPU temps sit on the sand.
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

/* ---------------------------------------------------------------- bowl */

#define N_FISH          6           /* pet fish that always live in the tank */
#define SURFACE_Y       62.0        /* water line */
#define SAND_Y          372.0       /* top of the sand */
#define SWIM_R          196.0       /* fish stay inside this radius */
#define MAX_BUBBLES     500
#define TOKENS_PER_FLAKE 25.0      /* one food flake per this many generated tokens */
#define MAX_FLAKES      700
#define N_WEEDS         6
#define N_PEBBLES       26

typedef struct {
    double x, y, heading, speed;    /* heading in radians, speed px/s */
    double turn;                    /* current turn rate */
    double tail;                    /* tail wag phase */
    double size;
    double gulp;                    /* 1 right after eating, decays */
    int    species;
} fish;

typedef struct { double x, y, vx, vy, rot, spin, settled; int kind, alive; } flake;

typedef struct { double x, y, r, wob, vy; int alive; } bubble;

static fish   fishes[N_FISH];
static bubble bubbles[MAX_BUBBLES];
static flake  flakes[MAX_FLAKES];
static double flake_acc[2];
static double bub_acc[3];
static double weed_x[N_WEEDS], weed_h[N_WEEDS], weed_p[N_WEEDS];
static double peb_x[N_PEBBLES], peb_y[N_PEBBLES], peb_r[N_PEBBLES], peb_c[N_PEBBLES];
static double snail_x = 150;

static double frand(void) { return rand() / (double)RAND_MAX; }

static const rgb GPU_BLUE   = { 0.20, 0.55, 1.0 };  /* ZOTAC stats */
static const rgb GPU_ORANGE = { 1.0, 0.52, 0.10 };  /* TUF stats */

static rgb fish_color(int species)
{
    static const rgb pal[N_FISH] = {
        { 0.20, 0.55, 1.00 },       /* blue tang */
        { 1.00, 0.52, 0.10 },       /* goldfish */
        { 1.00, 0.84, 0.15 },       /* yellow tang */
        { 0.95, 0.25, 0.25 },       /* red */
        { 0.65, 0.40, 1.00 },       /* purple */
        { 1.00, 0.60, 0.20 },       /* second goldfish */
    };
    return pal[species % N_FISH];
}

/* Flake food colours */
static rgb flake_color(int kind)
{
    static const rgb pal[4] = { { 0.85, 0.25, 0.20 }, { 0.95, 0.80, 0.30 }, { 0.45, 0.70, 0.25 }, { 0.80, 0.60, 0.40 } };
    return pal[kind & 3];
}

static void init_bowl(void)
{
    for (int i = 0; i < N_WEEDS; i++) {
        weed_x[i] = 110 + i * 52 + (frand() - 0.5) * 24;
        weed_h[i] = 70 + frand() * 90;
        weed_p[i] = frand() * 6;
    }
    for (int i = 0; i < N_PEBBLES; i++) {
        peb_x[i] = 70 + frand() * 340;
        peb_y[i] = SAND_Y + 16 + frand() * 50;
        peb_r[i] = 3 + frand() * 6;
        peb_c[i] = frand();
    }
    static const double sizes[N_FISH] = { 1.15, 1.0, 0.85, 0.75, 0.9, 0.8 };
    for (int i = 0; i < N_FISH; i++) {
        fish *f = &fishes[i];
        f->species = i;
        f->size = sizes[i];
        f->x = 130 + frand() * 220;
        f->y = 130 + frand() * 190;
        f->heading = frand() < 0.5 ? 0 : M_PI;
        f->speed = 30;
        f->tail = frand() * 6;
    }
}

static void spawn_bubble(double x, double y, double r)
{
    for (int i = 0; i < MAX_BUBBLES; i++)
        if (!bubbles[i].alive) {
            bubbles[i] = (bubble){ x, y, r, frand() * 6, -(35 + frand() * 25), 1 };
            return;
        }
}

static void spawn_flake(void)
{
    for (int i = 0; i < MAX_FLAKES; i++)
        if (!flakes[i].alive) {
            flakes[i] = (flake){ 90 + frand() * 300, SURFACE_Y + 4, (frand() - 0.5) * 30, 8 + frand() * 14,
                                 frand() * M_PI, (frand() - 0.5) * 4, 0, rand() & 3, 1 };
            return;
        }
}

/* Nearest free-falling flake of this fish's server, if any is close enough to chase */
static int nearest_flake(const fish *f)
{
    int best = -1;
    double bd = 230;
    for (int i = 0; i < MAX_FLAKES; i++) {
        const flake *k = &flakes[i];
        if (!k->alive || k->settled > 0)
            continue;
        if (k->y < SURFACE_Y + 35 || k->y > SAND_Y - 20 || hypot(k->x - SIZE / 2.0, k->y - SIZE / 2.0) > SWIM_R - 25)
            continue;                               /* out of reach: at the surface, on the sand, at the glass */
        double d = hypot(k->x - f->x, k->y - f->y);
        if (d < bd) {
            bd = d;
            best = i;
        }
    }
    return best;
}

static void simulate(const stats *s, double dt, double t)
{
    const double c = SIZE / 2.0;

    int food = 0;
    for (int i = 0; i < MAX_FLAKES; i++)
        food += flakes[i].alive && flakes[i].settled == 0;

    for (int i = 0; i < N_FISH; i++) {
        fish *f = &fishes[i];

        /* Lazy cruising when idle, livelier while food is falling */
        double want_speed = (s->tok_s < 1 ? 18 : 32) + (food ? 45 : 0) * (0.6 + 0.4 * f->size);
        f->speed += (want_speed - f->speed) * fmin(1, dt * 1.5);
        f->tail += dt * (4 + f->speed * 0.12);
        f->gulp *= exp(-dt * 6);

        double dx = f->x - c, dy = f->y - c, r = hypot(dx, dy);
        int k;
        f->turn += (frand() - 0.5) * 3.0 * dt;
        f->turn *= exp(-dt * 1.2);

        {
            if (r < SWIM_R - 20 && (k = nearest_flake(f)) >= 0) {
            /* Chase the nearest flake of our server's food and eat it */
            flake *fl = &flakes[k];
            double dir = cos(f->heading) >= 0 ? 1 : -1;
            double mx = f->x + dir * 22 * f->size, my = f->y;
            f->turn += remainder(atan2(fl->y - my, fl->x - mx) - f->heading, 2 * M_PI) * 5.0 * dt;
            if (hypot(fl->x - mx, fl->y - my) < 12) {
                fl->alive = 0;
                f->gulp = 1;
                if (frand() < 0.35)
                    spawn_bubble(mx, my - 4, 1.5 + frand());
            }
            }
            if (r > SWIM_R - 30 || f->y > SAND_Y - 40 || f->y < SURFACE_Y + 45) {
                /* Steer back inside the bowl (also brings new fish in from the edge) */
                double to_center = atan2(c + 10 - f->y, c - f->x);
                double diff = remainder(to_center - f->heading, 2 * M_PI);
                f->turn += diff * (r > SWIM_R ? 6.0 : 3.5) * dt;
            }
        }
        /* Keep a little distance from other fish */
        for (int j = 0; j < N_FISH; j++) {
            if (j == i)
                continue;
            double ex = f->x - fishes[j].x, ey = f->y - fishes[j].y, d = hypot(ex, ey);
            if (d < 50 && d > 0.1) {
                double away = atan2(ey, ex);
                f->turn += remainder(away - f->heading, 2 * M_PI) * (50 - d) / 50 * 1.5 * dt;
            }
        }
        f->turn = fmax(-2.2, fmin(2.2, f->turn));
        f->heading += f->turn * dt;
        /* Fish mostly swim level */
        double level = cos(f->heading) >= 0 ? 0 : M_PI;
        f->heading += remainder(level - f->heading, 2 * M_PI) * 0.35 * dt;
        f->x += cos(f->heading) * f->speed * dt;
        f->y += sin(f->heading) * f->speed * dt;
        f->y = fmax(SURFACE_Y + 22, fmin(SAND_Y - 14, f->y));
    }

    /* Food: tokens from every server sprinkle in anywhere on the surface */
    flake_acc[0] += fmin(s->tok_s, 3000) / TOKENS_PER_FLAKE * dt;
    while (flake_acc[0] >= 1) {
        spawn_flake();
        flake_acc[0] -= 1;
    }
    for (int i = 0; i < MAX_FLAKES; i++) {
        flake *k = &flakes[i];
        if (!k->alive)
            continue;
        if (k->settled > 0) {
            k->settled += dt;
            if (k->settled > 5)
                k->alive = 0;
            continue;
        }
        k->vx *= exp(-dt * 0.9);
        k->x += (k->vx + 16 * sin(t * 1.3 + i * 0.7)) * dt;
        k->y += k->vy * dt;
        k->vy = fmin(k->vy + 20 * dt, 32);
        k->rot += k->spin * dt;
        double ground = SAND_Y + 8 + 6 * sin(k->x * 0.03 + 1);
        if (k->y >= ground) {
            k->y = ground;
            k->settled = 0.001;
        }
    }

    bub_acc[2] += (s->tok_s < 1 ? 3.0 : 1.0) * dt;
    while (bub_acc[2] >= 1) {
        spawn_bubble(c + 118 + (frand() - 0.5) * 6, SAND_Y + 4, 2 + frand() * 3);
        bub_acc[2] -= 1;
    }
    for (int i = 0; i < MAX_BUBBLES; i++) {
        bubble *b = &bubbles[i];
        if (!b->alive)
            continue;
        b->y += b->vy * dt;
        b->vy -= 12 * dt;
        b->x += sin(t * 3 + b->wob) * 12 * dt;
        b->r += dt * 0.6;
        if (b->y < SURFACE_Y + 4)
            b->alive = 0;
    }

    snail_x += dt * 2.2;
    if (snail_x > 330)
        snail_x = 150;
}

/* ---------------------------------------------------------------- render */

static void set_rgb(cairo_t *cr, rgb c) { cairo_set_source_rgb(cr, c.r, c.g, c.b); }

static void text_center(cairo_t *cr, double x, double y, double size, int bold, rgb c, double a, const char *s)
{
    cairo_text_extents_t ext;
    cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL,
                           bold ? CAIRO_FONT_WEIGHT_BOLD : CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, size);
    cairo_text_extents(cr, s, &ext);
    double tx = x - ext.width / 2 - ext.x_bearing, ty = y - ext.height / 2 - ext.y_bearing;
    cairo_move_to(cr, tx + 2, ty + 2);
    cairo_set_source_rgba(cr, 0, 0.05, 0.1, 0.45 * a);
    cairo_show_text(cr, s);
    cairo_move_to(cr, tx, ty);
    cairo_set_source_rgba(cr, c.r, c.g, c.b, a);
    cairo_show_text(cr, s);
}

typedef struct { double zotac, tuf, tok; } shown_t;

static void draw_fish(cairo_t *cr, const fish *f)
{
    rgb col = fish_color(f->species);
    rgb dark = lerp(col, (rgb){ 0, 0, 0.1 }, 0.45);
    rgb light = lerp(col, (rgb){ 1, 1, 1 }, 0.45);
    double dir = cos(f->heading) >= 0 ? 1 : -1;
    double tilt = atan2(sin(f->heading), fabs(cos(f->heading))) * 0.6;
    double wag = sin(f->tail) * 0.35;

    cairo_save(cr);
    cairo_translate(cr, f->x, f->y);
    cairo_scale(cr, dir * f->size * (1 + 0.12 * f->gulp), f->size * (1 + 0.12 * f->gulp));
    cairo_rotate(cr, tilt * dir);
    cairo_push_group(cr);

    /* Tail, wagging */
    cairo_save(cr);
    cairo_translate(cr, -18, 0);
    cairo_rotate(cr, wag);
    cairo_move_to(cr, 2, 0);
    cairo_curve_to(cr, -8, -4, -16, -16, -20, -14);
    cairo_curve_to(cr, -15, -5, -15, 5, -20, 14);
    cairo_curve_to(cr, -16, 16, -8, 4, 2, 0);
    set_rgb(cr, dark);
    cairo_fill(cr);
    cairo_restore(cr);

    /* Dorsal and pelvic fins */
    cairo_move_to(cr, -6, -12);
    cairo_curve_to(cr, -2, -22, 8, -20, 10, -11);
    set_rgb(cr, dark);
    cairo_fill(cr);
    cairo_move_to(cr, -2, 11);
    cairo_curve_to(cr, 0, 18, 6, 18, 8, 11);
    cairo_fill(cr);

    /* Body with a lit belly */
    cairo_save(cr);
    cairo_scale(cr, 1, 0.58);
    cairo_arc(cr, 0, 0, 22, 0, 2 * M_PI);
    cairo_restore(cr);
    cairo_pattern_t *g = cairo_pattern_create_linear(0, -13, 0, 13);
    cairo_pattern_add_color_stop_rgb(g, 0, dark.r, dark.g, dark.b);
    cairo_pattern_add_color_stop_rgb(g, 0.45, col.r, col.g, col.b);
    cairo_pattern_add_color_stop_rgb(g, 1, light.r, light.g, light.b);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);

    /* Stripe, eye and a side fin that flaps with the tail */
    cairo_move_to(cr, 6, -11);
    cairo_curve_to(cr, 3, -3, 3, 3, 6, 11);
    cairo_set_line_width(cr, 2.5);
    cairo_set_source_rgba(cr, 1, 1, 1, 0.55);
    cairo_stroke(cr);
    cairo_arc(cr, 13, -3, 3.6, 0, 2 * M_PI);
    cairo_set_source_rgb(cr, 1, 1, 1);
    cairo_fill(cr);
    cairo_arc(cr, 14, -3, 2.1, 0, 2 * M_PI);
    cairo_set_source_rgb(cr, 0.05, 0.05, 0.1);
    cairo_fill(cr);
    cairo_save(cr);
    cairo_translate(cr, 2, 3);
    cairo_rotate(cr, 0.5 + sin(f->tail * 1.3) * 0.4);
    cairo_move_to(cr, 0, 0);
    cairo_curve_to(cr, -6, 2, -9, 7, -7, 9);
    cairo_curve_to(cr, -3, 7, -1, 4, 0, 0);
    set_rgb(cr, light);
    cairo_fill(cr);
    cairo_restore(cr);

    cairo_pop_group_to_source(cr);
    cairo_paint(cr);
    cairo_restore(cr);
}

static void draw_snail(cairo_t *cr, double x, double t)
{
    double y = SAND_Y + 12;
    double stretch = 1 + 0.08 * sin(t * 1.5);
    cairo_save(cr);
    cairo_translate(cr, x, y);
    /* body */
    cairo_move_to(cr, -12 * stretch, 0);
    cairo_curve_to(cr, -12 * stretch, -5, 14 * stretch, -6, 16 * stretch, -2);
    cairo_line_to(cr, 20 * stretch, -12);
    cairo_line_to(cr, 17 * stretch, -1);
    cairo_line_to(cr, 16 * stretch, 1);
    cairo_close_path(cr);
    cairo_set_source_rgb(cr, 0.80, 0.72, 0.55);
    cairo_fill(cr);
    cairo_arc(cr, 20 * stretch, -12, 1.8, 0, 2 * M_PI);
    cairo_set_source_rgb(cr, 0.2, 0.15, 0.1);
    cairo_fill(cr);
    /* shell spiral */
    cairo_pattern_t *g = cairo_pattern_create_radial(-1, -12, 2, 0, -10, 12);
    cairo_pattern_add_color_stop_rgb(g, 0, 0.85, 0.55, 0.30);
    cairo_pattern_add_color_stop_rgb(g, 1, 0.45, 0.22, 0.12);
    cairo_arc(cr, 0, -10, 11, 0, 2 * M_PI);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    cairo_new_path(cr);
    for (double a = 0; a < 4 * M_PI; a += 0.2) {
        double rr = 1 + a * 0.75;
        cairo_line_to(cr, rr * cos(a), -10 + rr * sin(a));
    }
    cairo_set_line_width(cr, 1.3);
    cairo_set_source_rgba(cr, 0.3, 0.12, 0.05, 0.8);
    cairo_stroke(cr);
    cairo_restore(cr);
}

static void render(cairo_t *cr, const stats *s, const shown_t *sh, double t)
{
    const double c = SIZE / 2.0;
    double total_w = s->power[0] + s->power[1];
    int idle = s->tok_s < 1 && s->running == 0;
    char txt[64];

    /* Water */
    {
        cairo_pattern_t *g = cairo_pattern_create_linear(0, SURFACE_Y, 0, SAND_Y);
        cairo_pattern_add_color_stop_rgb(g, 0, 0.10, 0.45, 0.62);
        cairo_pattern_add_color_stop_rgb(g, 1, 0.02, 0.14, 0.26);
        cairo_set_source(cr, g);
        cairo_paint(cr);
        cairo_pattern_destroy(g);
        /* Air above the water line */
        cairo_rectangle(cr, 0, 0, SIZE, SURFACE_Y);
        cairo_set_source_rgb(cr, 0.03, 0.06, 0.10);
        cairo_fill(cr);
    }

    /* Light rays swaying down from the surface */
    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    for (int i = 0; i < 5; i++) {
        double x0 = 90 + i * 75 + 18 * sin(t * 0.3 + i);
        double w = 26 + 10 * sin(t * 0.5 + i * 2);
        double lean = 60 + 25 * sin(t * 0.2 + i);
        cairo_pattern_t *g = cairo_pattern_create_linear(0, SURFACE_Y, 0, SAND_Y);
        cairo_pattern_add_color_stop_rgba(g, 0, 0.6, 0.9, 1.0, 0.10);
        cairo_pattern_add_color_stop_rgba(g, 1, 0.6, 0.9, 1.0, 0);
        cairo_move_to(cr, x0, SURFACE_Y);
        cairo_line_to(cr, x0 + w, SURFACE_Y);
        cairo_line_to(cr, x0 + w * 2.2 + lean, SAND_Y);
        cairo_line_to(cr, x0 + lean, SAND_Y);
        cairo_close_path(cr);
        cairo_set_source(cr, g);
        cairo_fill(cr);
        cairo_pattern_destroy(g);
    }
    /* Caustic shimmer near the surface */
    cairo_set_line_width(cr, 2);
    for (int i = 0; i < 7; i++) {
        double y = SURFACE_Y + 14 + i * 13;
        cairo_new_path(cr);
        for (double x = 20; x <= SIZE - 20; x += 8) {
            double yy = y + 4 * sin(x * 0.045 + t * (1.1 + i * 0.13) + i) + 3 * sin(x * 0.11 - t * 1.7);
            cairo_line_to(cr, x, yy);
        }
        cairo_set_source_rgba(cr, 0.7, 0.95, 1.0, 0.07 * (1 - i / 7.0));
        cairo_stroke(cr);
    }
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);

    /* Seaweed, swaying */
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    for (int i = 0; i < N_WEEDS; i++) {
        double bx = weed_x[i], h = weed_h[i];
        cairo_new_path(cr);
        cairo_move_to(cr, bx, SAND_Y + 14);
        for (int k = 1; k <= 12; k++) {
            double u = k / 12.0;
            double sway = sin(t * 1.1 + weed_p[i] + u * 2.2) * 14 * u;
            cairo_line_to(cr, bx + sway, SAND_Y + 14 - h * u);
        }
        cairo_set_line_width(cr, 7);
        cairo_set_source_rgb(cr, 0.08, 0.42 + 0.1 * (i % 2), 0.22);
        cairo_stroke_preserve(cr);
        cairo_set_line_width(cr, 2);
        cairo_set_source_rgba(cr, 0.4, 0.85, 0.45, 0.5);
        cairo_stroke(cr);
    }

    /* Sand and pebbles */
    {
        cairo_new_path(cr);
        cairo_move_to(cr, 0, SAND_Y + 10);
        for (double x = 0; x <= SIZE; x += 16)
            cairo_line_to(cr, x, SAND_Y + 6 * sin(x * 0.03 + 1) + 4);
        cairo_line_to(cr, SIZE, SIZE);
        cairo_line_to(cr, 0, SIZE);
        cairo_close_path(cr);
        cairo_pattern_t *g = cairo_pattern_create_linear(0, SAND_Y, 0, SIZE);
        cairo_pattern_add_color_stop_rgb(g, 0, 0.78, 0.66, 0.45);
        cairo_pattern_add_color_stop_rgb(g, 1, 0.40, 0.31, 0.20);
        cairo_set_source(cr, g);
        cairo_fill(cr);
        cairo_pattern_destroy(g);
        for (int i = 0; i < N_PEBBLES; i++) {
            rgb pc = lerp((rgb){ 0.45, 0.42, 0.40 }, (rgb){ 0.75, 0.60, 0.50 }, peb_c[i]);
            cairo_save(cr);
            cairo_translate(cr, peb_x[i], peb_y[i]);
            cairo_scale(cr, 1.4, 1);
            cairo_arc(cr, 0, 0, peb_r[i], 0, 2 * M_PI);
            cairo_restore(cr);
            set_rgb(cr, pc);
            cairo_fill(cr);
        }
    }

    /* Bubbler stone */
    cairo_save(cr);
    cairo_translate(cr, c + 118, SAND_Y + 8);
    cairo_scale(cr, 1.8, 1);
    cairo_arc(cr, 0, 0, 8, 0, 2 * M_PI);
    cairo_restore(cr);
    cairo_set_source_rgb(cr, 0.35, 0.36, 0.40);
    cairo_fill(cr);

    draw_snail(cr, snail_x, t);

    /* Food flakes */
    for (int i = 0; i < MAX_FLAKES; i++) {
        const flake *k = &flakes[i];
        if (!k->alive)
            continue;
        rgb col = flake_color(k->kind);
        double a = k->settled > 0 ? clamp01(1 - (k->settled - 3) / 2) : 1;
        cairo_save(cr);
        cairo_translate(cr, k->x, k->y);
        cairo_rotate(cr, k->rot);
        cairo_move_to(cr, -3.2, -1.2);
        cairo_line_to(cr, 1.5, -2.4);
        cairo_line_to(cr, 3.4, 1.0);
        cairo_line_to(cr, -1.2, 2.2);
        cairo_close_path(cr);
        cairo_set_source_rgba(cr, col.r, col.g, col.b, a);
        cairo_fill(cr);
        cairo_restore(cr);
    }

    for (int i = 0; i < N_FISH; i++)
        draw_fish(cr, &fishes[i]);

    /* Bubbles */
    for (int i = 0; i < MAX_BUBBLES; i++) {
        const bubble *b = &bubbles[i];
        if (!b->alive)
            continue;
        cairo_new_path(cr);
        cairo_arc(cr, b->x, b->y, b->r, 0, 2 * M_PI);
        cairo_set_source_rgba(cr, 0.8, 0.95, 1.0, 0.12);
        cairo_fill_preserve(cr);
        cairo_set_line_width(cr, 1.1);
        cairo_set_source_rgba(cr, 0.85, 0.97, 1.0, 0.7);
        cairo_stroke(cr);
        cairo_arc(cr, b->x - b->r * 0.35, b->y - b->r * 0.35, b->r * 0.28, 0, 2 * M_PI);
        cairo_set_source_rgba(cr, 1, 1, 1, 0.8);
        cairo_fill(cr);
    }

    /* Water surface with a moving highlight */
    cairo_new_path(cr);
    for (double x = 0; x <= SIZE; x += 6)
        cairo_line_to(cr, x, SURFACE_Y + 2.5 * sin(x * 0.05 + t * 2) + 1.5 * sin(x * 0.13 - t * 3));
    cairo_set_line_width(cr, 2.5);
    cairo_set_source_rgba(cr, 0.75, 0.95, 1.0, 0.8);
    cairo_stroke(cr);

    /* Tokens/sec in the air above the water, stats on the sand */
    if (idle) {
        text_center(cr, c, 36, 22, 1, (rgb){ 0.6, 0.75, 0.85 }, 1, "IDLE");
    } else {
        snprintf(txt, sizeof(txt), "%.0f tok/s", sh->tok);
        text_center(cr, c, 36, 28, 1, WHITE, 1, txt);
    }
    snprintf(txt, sizeof(txt), "%.0f W", total_w);
    text_center(cr, c, SAND_Y + 40, 28, 1, WHITE, 1, txt);
    snprintf(txt, sizeof(txt), "%d\xC2\xB0", s->temp[0]);
    text_center(cr, c - 44, SAND_Y + 74, 22, 1, GPU_BLUE, 1, txt);
    snprintf(txt, sizeof(txt), "%d\xC2\xB0", s->temp[1]);
    text_center(cr, c + 44, SAND_Y + 74, 22, 1, GPU_ORANGE, 1, txt);

    /* Glass bowl rim and reflection */
    cairo_new_path(cr);
    cairo_arc(cr, c, c, SIZE / 2.0 - 3, 0, 2 * M_PI);
    cairo_set_line_width(cr, 5);
    cairo_set_source_rgba(cr, 0.8, 0.95, 1.0, 0.25);
    cairo_stroke(cr);
    cairo_new_path(cr);
    cairo_arc(cr, c, c, SIZE / 2.0 - 22, M_PI * 1.12, M_PI * 1.38);
    cairo_set_line_width(cr, 9);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    cairo_set_source_rgba(cr, 1, 1, 1, 0.14);
    cairo_stroke(cr);
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
    init_bowl();

    surf = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cr = cairo_create(surf);
    tj = tj3Init(TJINIT_COMPRESS);
    tj3Set(tj, TJPARAM_QUALITY, JPEG_QUALITY);
    tj3Set(tj, TJPARAM_SUBSAMP, TJSAMP_420);

    if (bench) {
        struct { double tok0, tok1; int run0, run1; double w0, w1; const char *png; } scenes[] = {
            { 723, 884, 4, 4, 470, 460, "fishbowl_preview.png" },
            { 420, 190, 3, 1, 330, 250, "fishbowl_mid.png" },
            { 0, 0, 0, 0, 25, 30, "fishbowl_idle.png" },
        };
        for (int k = 0; k < 3; k++) {
            memset(bubbles, 0, sizeof(bubbles));
            memset(flakes, 0, sizeof(flakes));
            s.tok_port[0] = scenes[k].tok0; s.tok_port[1] = scenes[k].tok1;
            s.tok_s = s.tok_port[0] + s.tok_port[1]; s.running = scenes[k].run0 + scenes[k].run1;
            s.power[0] = scenes[k].w0; s.power[1] = scenes[k].w1; s.temp[0] = 54; s.temp[1] = 71;
            sh.tok = s.tok_s;
            int n = 200;
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
                /* scale a copy each second; scaling s itself would compound */
                static stats demo_base;
                demo_poll(&demo_base, t - t0);
                s = demo_base;
                s.tok_port[0] *= 6; s.tok_port[1] *= 6;
                s.tok_s = s.tok_port[0] + s.tok_port[1];
                if (s.tok_s < 60) s.tok_s = s.tok_port[0] = s.tok_port[1] = 0;
                s.running = s.tok_s < 1 ? 0 : 1 + (int)(s.tok_s / 250);
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
