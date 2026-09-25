/*
 * synapse: generated tokens fire signals through a glowing neural network on the
 * iCUE LINK AIO pump LCD. Left half (blue) is fed by the ZOTAC's vLLM server, right
 * half (orange) by the TUF's. Signals hop inward layer by layer into the core,
 * which shows tokens/sec. Run with --demo to simulate data, --bench to benchmark.
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
#define FPS             20
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

static const rgb BLUE   = { 61 / 255.0, 174 / 255.0, 233 / 255.0 };
static const rgb ORANGE = { 233 / 255.0, 120 / 255.0, 61 / 255.0 };
static const rgb WHITE  = { 240 / 255.0, 240 / 255.0, 245 / 255.0 };
static const rgb DIM    = { 110 / 255.0, 115 / 255.0, 125 / 255.0 };
static const rgb TRACK  = { 38 / 255.0, 40 / 255.0, 48 / 255.0 };
static const rgb BG     = { 6 / 255.0, 7 / 255.0, 12 / 255.0 };

typedef struct {
    double load[N_GPUS];        /* 0..1 */
    double power[N_GPUS];       /* W */
    int    temp[N_GPUS];        /* C */
    double tok_s;               /* all servers */
    double tok_port[2];         /* per server: [0] ZOTAC's vLLM, [1] TUF's vLLM */
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
            s->tok_port[i] *= 0.5;
            s->tok_s += s->tok_port[i];
            continue;
        }
        tokens   = metric_sum(buf, "vllm:generation_tokens_total");
        running += (int)metric_sum(buf, "vllm:num_requests_running");
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

/* ---------------------------------------------------------------- network */

#define CORE_R          70.0        /* central output node radius (px) */
#define N_LAYERS        3
#define MAX_NODES       64
#define MAX_EDGES       256
#define MAX_PULSES      1200
#define TOKENS_PER_P    2.0         /* one signal per two generated tokens */
#define AMBIENT_RATE    2.5         /* signals/sec when idle */
#define PULSE_SPEED     520.0       /* px/sec along an edge */

static const double layer_r[N_LAYERS]   = { 206, 158, 112 };
static const int    layer_n[N_LAYERS]   = { 22, 16, 12 };

typedef struct {
    double x, y;
    double act;             /* activation 0..1, decays */
    int    layer;           /* 0 outer .. N_LAYERS-1 inner, N_LAYERS = core */
    int    side;            /* 0 left (ZOTAC), 1 right (TUF) */
    int    out[3];          /* outgoing edge indices */
    int    n_out;
} node;

typedef struct { int a, b; double len, heat; } edge;

typedef struct {
    int    edge;
    double pos;             /* 0..1 along the edge */
    int    src;             /* 0 ZOTAC, 1 TUF, 2 ambient */
    int    alive;
} pulse;

static node  nodes[MAX_NODES];
static edge  edges[MAX_EDGES];
static pulse pulses[MAX_PULSES];
static int   n_nodes, n_edges, core_node;
static double core_flash[3];
static double spawn_acc[3];

static double frand(void) { return rand() / (double)RAND_MAX; }

static int add_edge(int a, int b)
{
    if (n_edges >= MAX_EDGES || nodes[a].n_out >= 3)
        return -1;
    edge *e = &edges[n_edges];
    e->a = a;
    e->b = b;
    e->len = hypot(nodes[b].x - nodes[a].x, nodes[b].y - nodes[a].y);
    nodes[a].out[nodes[a].n_out++] = n_edges;
    return n_edges++;
}

/* Build a fixed, slightly irregular layered network; same seed every run */
static void build_network(void)
{
    const double c = SIZE / 2.0;
    int first[N_LAYERS + 1];

    srand(1337);
    for (int l = 0; l < N_LAYERS; l++) {
        first[l] = n_nodes;
        double off = frand() * 2 * M_PI;
        for (int i = 0; i < layer_n[l]; i++) {
            node *n = &nodes[n_nodes++];
            double a = off + (i + (frand() - 0.5) * 0.5) * 2 * M_PI / layer_n[l];
            double r = layer_r[l] + (frand() - 0.5) * 16;
            n->x = c + r * cos(a);
            n->y = c + r * sin(a);
            n->layer = l;
            n->side = n->x >= c;
        }
    }
    first[N_LAYERS] = core_node = n_nodes;
    nodes[n_nodes++] = (node){ .x = c, .y = c, .layer = N_LAYERS };

    /* Each node connects to its 2 nearest nodes in the next layer in (inner layer: the core) */
    for (int l = 0; l < N_LAYERS; l++) {
        for (int i = first[l]; i < first[l + 1]; i++) {
            if (l == N_LAYERS - 1) {
                add_edge(i, core_node);
                continue;
            }
            int best[2] = { -1, -1 };
            double bd[2] = { 1e9, 1e9 };
            for (int j = first[l + 1]; j < first[l + 2]; j++) {
                double d = hypot(nodes[j].x - nodes[i].x, nodes[j].y - nodes[i].y);
                if (d < bd[0]) { bd[1] = bd[0]; best[1] = best[0]; bd[0] = d; best[0] = j; }
                else if (d < bd[1]) { bd[1] = d; best[1] = j; }
            }
            for (int k = 0; k < 2; k++)
                if (best[k] >= 0)
                    add_edge(i, best[k]);
        }
    }
    srand((unsigned)time(NULL));
}

static void fire(int src)
{
    /* Start on a random outer node on this server's side */
    int tries = 0, n;
    do {
        n = rand() % layer_n[0];
    } while (src < 2 && nodes[n].side != src && ++tries < 50);
    if (!nodes[n].n_out)
        return;
    for (int i = 0; i < MAX_PULSES; i++) {
        if (pulses[i].alive)
            continue;
        pulses[i] = (pulse){ .edge = nodes[n].out[rand() % nodes[n].n_out], .pos = 0, .src = src, .alive = 1 };
        nodes[n].act = 1;
        return;
    }
}

static void simulate(const stats *s, double dt)
{
    double rates[3] = { s->tok_port[0] / TOKENS_PER_P, s->tok_port[1] / TOKENS_PER_P, 0 };
    if (s->tok_s < 1)
        rates[2] = AMBIENT_RATE;

    for (int k = 0; k < 3; k++) {
        spawn_acc[k] += rates[k] * dt;
        while (spawn_acc[k] >= 1) {
            fire(k);
            spawn_acc[k] -= 1;
        }
        core_flash[k] *= exp(-dt * 5);
    }
    for (int i = 0; i < n_nodes; i++)
        nodes[i].act *= exp(-dt * 4);
    for (int i = 0; i < n_edges; i++)
        edges[i].heat *= exp(-dt * 3);

    for (int i = 0; i < MAX_PULSES; i++) {
        pulse *p = &pulses[i];
        if (!p->alive)
            continue;
        edge *e = &edges[p->edge];
        p->pos += PULSE_SPEED * dt / e->len;
        e->heat = fmin(1, e->heat + dt * 6);
        if (p->pos < 1)
            continue;
        /* Arrived: light the node, then hop on or finish at the core */
        node *n = &nodes[e->b];
        n->act = 1;
        if (e->b == core_node || !n->n_out) {
            core_flash[p->src] = fmin(1, core_flash[p->src] + 0.25);
            p->alive = 0;
        } else {
            p->edge = n->out[rand() % n->n_out];
            p->pos = 0;
        }
    }
}

/* ---------------------------------------------------------------- render */

static void set_rgb(cairo_t *cr, rgb c) { cairo_set_source_rgb(cr, c.r, c.g, c.b); }

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

static rgb src_color(int src)
{
    static const rgb AMBIENT = { 0.55, 0.60, 0.75 };
    return src == 0 ? BLUE : src == 1 ? ORANGE : AMBIENT;
}

/* Glows are pre-rendered once per color as small sprites and stamped, instead of
 * building a radial gradient for every neuron and signal on every frame. */
#define SPRITE_R        32
enum { SPR_BLUE, SPR_ORANGE, SPR_BLUE_HOT, SPR_ORANGE_HOT, SPR_AMBIENT_HOT, N_SPRITES };
static cairo_surface_t *sprites[N_SPRITES];

static void build_sprites(void)
{
    static const rgb AMBIENT = { 0.55, 0.60, 0.75 };
    const rgb W = { 1, 1, 1 };
    rgb cols[N_SPRITES] = { BLUE, ORANGE, lerp(BLUE, W, 0.5), lerp(ORANGE, W, 0.5), lerp(AMBIENT, W, 0.5) };

    for (int i = 0; i < N_SPRITES; i++) {
        sprites[i] = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SPRITE_R * 2, SPRITE_R * 2);
        cairo_t *sc = cairo_create(sprites[i]);
        cairo_pattern_t *g = cairo_pattern_create_radial(SPRITE_R, SPRITE_R, 0, SPRITE_R, SPRITE_R, SPRITE_R);
        cairo_pattern_add_color_stop_rgba(g, 0.0, cols[i].r, cols[i].g, cols[i].b, 1);
        cairo_pattern_add_color_stop_rgba(g, 1.0, cols[i].r, cols[i].g, cols[i].b, 0);
        cairo_set_source(sc, g);
        cairo_paint(sc);
        cairo_pattern_destroy(g);
        cairo_destroy(sc);
    }
}

static void glow_dot(cairo_t *cr, double x, double y, double r, int sprite, double a)
{
    double k = r / SPRITE_R;
    cairo_save(cr);
    cairo_translate(cr, x - r, y - r);
    cairo_scale(cr, k, k);
    cairo_rectangle(cr, 0, 0, SPRITE_R * 2, SPRITE_R * 2);   /* keep compositing to the sprite's box */
    cairo_clip(cr);
    cairo_set_source_surface(cr, sprites[sprite], 0, 0);
    cairo_paint_with_alpha(cr, a);
    cairo_restore(cr);
}

static void render(cairo_t *cr, const stats *s, const shown_t *sh, double t)
{
    const double c = SIZE / 2.0;
    double total_w = s->power[0] + s->power[1];
    double heat = clamp01(total_w / POWER_MAX);
    rgb core = heat_color(heat);
    char txt[64];

    set_rgb(cr, BG);
    cairo_paint(cr);

    /* Edges: faint wiring that warms up while signals travel on it */
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    for (int i = 0; i < n_edges; i++) {
        const edge *e = &edges[i];
        const node *a = &nodes[e->a], *b = &nodes[e->b];
        rgb col = lerp(TRACK, a->side ? ORANGE : BLUE, 0.25 + 0.55 * e->heat);
        cairo_set_source_rgba(cr, col.r, col.g, col.b, 0.35 + 0.5 * e->heat);
        cairo_set_line_width(cr, 1.2 + 1.3 * e->heat);
        cairo_move_to(cr, a->x, a->y);
        cairo_line_to(cr, b->x, b->y);
        cairo_stroke(cr);
    }

    /* Signals: bright heads with a short tail, additive */
    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    for (int i = 0; i < MAX_PULSES; i++) {
        const pulse *p = &pulses[i];
        if (!p->alive)
            continue;
        const edge *e = &edges[p->edge];
        const node *a = &nodes[e->a], *b = &nodes[e->b];
        double x = a->x + (b->x - a->x) * p->pos, y = a->y + (b->y - a->y) * p->pos;
        double tail = fmax(0, p->pos - 26 / e->len);
        double tx = a->x + (b->x - a->x) * tail, ty = a->y + (b->y - a->y) * tail;
        rgb col = src_color(p->src);
        cairo_set_source_rgba(cr, col.r, col.g, col.b, 0.8);
        cairo_set_line_width(cr, 2.5);
        cairo_move_to(cr, tx, ty);
        cairo_line_to(cr, x, y);
        cairo_stroke(cr);
        glow_dot(cr, x, y, 7, p->src == 0 ? SPR_BLUE_HOT : p->src == 1 ? SPR_ORANGE_HOT : SPR_AMBIENT_HOT, 0.9);
    }

    /* Neurons */
    for (int i = 0; i < n_nodes; i++) {
        const node *n = &nodes[i];
        if (i == core_node)
            continue;
        if (n->act > 0.02)
            glow_dot(cr, n->x, n->y, 8 + 14 * n->act, n->side ? SPR_ORANGE : SPR_BLUE, 0.7 * n->act);
    }
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    for (int i = 0; i < n_nodes; i++) {
        const node *n = &nodes[i];
        if (i == core_node)
            continue;
        rgb col = lerp(lerp(TRACK, n->side ? ORANGE : BLUE, 0.45), (rgb){ 1, 1, 1 }, 0.7 * n->act);
        set_rgb(cr, col);
        cairo_arc(cr, n->x, n->y, 3.5 + 1.5 * n->act, 0, 2 * M_PI);
        cairo_fill(cr);
    }

    /* Core: glow that heats with power and flashes with arriving signals */
    {
        double flash = fmin(1, core_flash[0] + core_flash[1] + core_flash[2]);
        rgb fl = lerp(core, flash > 0 ? lerp(BLUE, ORANGE, core_flash[1] / (core_flash[0] + core_flash[1] + 1e-6)) : core, 0.4);
        double a = 0.25 + 0.35 * heat + 0.4 * flash;
        cairo_pattern_t *g = cairo_pattern_create_radial(c, c, CORE_R * 0.9, c, c, CORE_R * 1.9);
        cairo_pattern_add_color_stop_rgba(g, 0.0, fl.r, fl.g, fl.b, a);
        cairo_pattern_add_color_stop_rgba(g, 1.0, fl.r, fl.g, fl.b, 0);
        cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
        cairo_set_source(cr, g);
        cairo_paint(cr);
        cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
        cairo_pattern_destroy(g);

        cairo_set_line_width(cr, 3.0);
        cairo_new_sub_path(cr);
        cairo_arc(cr, c, c, CORE_R + 1.5, 0, 2 * M_PI);
        rgb ring = lerp(fl, (rgb){ 1, 1, 1 }, 0.3 + 0.4 * flash);
        cairo_set_source_rgba(cr, ring.r, ring.g, ring.b, 0.8 + 0.2 * sin(t * 2.7));
        cairo_stroke(cr);
        cairo_new_sub_path(cr);
        cairo_arc(cr, c, c, CORE_R, 0, 2 * M_PI);
        cairo_set_source_rgb(cr, 0.02, 0.02, 0.04);
        cairo_fill(cr);
    }

    if (s->tok_s < 0.5 && s->running == 0) {
        text_center(cr, c, c, 26, 1, DIM, "IDLE");
    } else {
        snprintf(txt, sizeof(txt), "%.0f", sh->tok);
        text_center(cr, c, c - 9, sh->tok >= 1000 ? 42 : 54, 1, WHITE, txt);
        text_center(cr, c, c + 30, 16, 1, core, "TOK/S");
    }

    /* Thin GPU load arcs at the rim */
    {
        double ring_r = SIZE / 2.0 - 4, ring_w = 6;
        cairo_set_line_cap(cr, CAIRO_LINE_CAP_BUTT);
        arc(cr, ring_r, ring_w, 95, 265, TRACK);
        arc(cr, ring_r, ring_w, 275, 445, TRACK);
        if (sh->zotac > 0.005)
            arc(cr, ring_r, ring_w, 265 - 170 * sh->zotac, 265, BLUE);
        if (sh->tuf > 0.005)
            arc(cr, ring_r, ring_w, 275, 275 + 170 * sh->tuf, ORANGE);
    }

    /* Darken the bottom so the stats stay readable over the network */
    {
        cairo_pattern_t *g = cairo_pattern_create_linear(0, c + 105, 0, c + 190);
        cairo_pattern_add_color_stop_rgba(g, 0.0, BG.r, BG.g, BG.b, 0);
        cairo_pattern_add_color_stop_rgba(g, 0.45, BG.r, BG.g, BG.b, 0.82);
        cairo_pattern_add_color_stop_rgba(g, 1.0, BG.r, BG.g, BG.b, 0.9);
        cairo_rectangle(cr, 0, c + 105, SIZE, SIZE);
        cairo_set_source(cr, g);
        cairo_fill(cr);
        cairo_pattern_destroy(g);
    }

    snprintf(txt, sizeof(txt), "%.0f W", total_w);
    text_center(cr, c, c + 138, 34, 1, WHITE, txt);
    snprintf(txt, sizeof(txt), "%d\xC2\xB0", s->temp[0]);
    text_center(cr, c - 72, c + 180, 26, 1, BLUE, txt);
    snprintf(txt, sizeof(txt), "%d\xC2\xB0", s->temp[1]);
    text_center(cr, c + 72, c + 180, 26, 1, ORANGE, txt);
    if (s->running) {
        snprintf(txt, sizeof(txt), "%d req", s->running);
        text_center(cr, c, c + 180, 18, 1, DIM, txt);
    }
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

    build_network();
    build_sprites();
    surf = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cr = cairo_create(surf);
    tj = tj3Init(TJINIT_COMPRESS);
    tj3Set(tj, TJPARAM_QUALITY, JPEG_QUALITY);
    tj3Set(tj, TJPARAM_SUBSAMP, TJSAMP_420);

    if (bench) {
        s.power[0] = 440; s.power[1] = 430; s.temp[0] = 54; s.temp[1] = 71;
        s.tok_port[0] = 120; s.tok_port[1] = 95; s.tok_s = 215; s.running = 5;
        sh.zotac = 0.95; sh.tuf = 0.9; sh.tok = 215;
        for (int i = 0; i < FPS * 10; i++)
            simulate(&s, 1.0 / FPS);
        size_t len = 0;
        int live = 0;
        double b0 = now_s();
        for (int i = 0; i < 300; i++) {
            simulate(&s, 1.0 / FPS);
            render(cr, &s, &sh, i / (double)FPS);
            cairo_surface_flush(surf);
            tj3Compress8(tj, cairo_image_surface_get_data(surf), SIZE, cairo_image_surface_get_stride(surf),
                         SIZE, TJPF_BGRX, &jpeg, &len);
        }
        for (int i = 0; i < MAX_PULSES; i++)
            live += pulses[i].alive;
        printf("%.2f ms/frame, jpeg %zu bytes, %d nodes, %d edges, %d signals\n",
               (now_s() - b0) * 1000 / 300, len, n_nodes, n_edges, live);
        cairo_surface_write_to_png(surf, "synapse_preview.png");
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
            } else {
                gpus_poll(&s);
                vllm_poll(&s, t);
            }
        }

        k = dt * 4 > 1 ? 1 : dt * 4;
        sh.zotac += (s.load[0] - sh.zotac) * k;
        sh.tuf   += (s.load[1] - sh.tuf) * k;
        sh.tok   += (s.tok_s - sh.tok) * k;

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
