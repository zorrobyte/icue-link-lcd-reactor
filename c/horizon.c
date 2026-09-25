/*
 * horizon: your LLM's actual output tokens fall into a black hole on the
 * iCUE LINK AIO pump LCD.
 *
 * Streamed text pieces are captured passively from the vLLM servers' responses on
 * loopback (libpcap). A legible subset spirals in as words: tangent to their orbit
 * at first, then turning toward the hole, stretching (spaghettification), heating
 * to white and redshifting out at the horizon. The rest of the real token flow is
 * drawn as an accretion disc of dust. Blue is the ZOTAC's server, orange the TUF's.
 * A lensed starfield, photon ring and tokens/sec complete it.
 *
 * Needs root (or CAP_NET_RAW) for capture and hidraw access.
 * Run with --demo to simulate data, --bench to benchmark rendering.
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
#include <pcap/pcap.h>
#include <pthread.h>
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

/* ---------------------------------------------------------------- token capture */

/*
 * A capture thread sniffs the vLLM servers' responses on loopback and pulls out the
 * streamed text pieces ("content", "reasoning_content", "text"). Each piece becomes a
 * word that falls into the black hole. Nothing is stored or sent anywhere.
 */
#define TOKQ_SIZE       4096
#define TOK_MAX_BYTES   28

typedef struct { char text[TOK_MAX_BYTES + 4]; int src; } tok_item;

static tok_item        tokq[TOKQ_SIZE];
static int             tokq_head, tokq_tail;
static pthread_mutex_t tokq_lock = PTHREAD_MUTEX_INITIALIZER;
static long            tokq_dropped;

static void tokq_push(const char *s, int src)
{
    pthread_mutex_lock(&tokq_lock);
    int next = (tokq_head + 1) % TOKQ_SIZE;
    if (next == tokq_tail) {                    /* full: drop the oldest */
        tokq_tail = (tokq_tail + 1) % TOKQ_SIZE;
        tokq_dropped++;
    }
    snprintf(tokq[tokq_head].text, sizeof(tokq[tokq_head].text), "%s", s);
    tokq[tokq_head].src = src;
    tokq_head = next;
    pthread_mutex_unlock(&tokq_lock);
}

static int tokq_pop(tok_item *out)
{
    int ok = 0;
    pthread_mutex_lock(&tokq_lock);
    if (tokq_tail != tokq_head) {
        *out = tokq[tokq_tail];
        tokq_tail = (tokq_tail + 1) % TOKQ_SIZE;
        ok = 1;
    }
    pthread_mutex_unlock(&tokq_lock);
    return ok;
}

static int tokq_len(void)
{
    pthread_mutex_lock(&tokq_lock);
    int n = (tokq_head - tokq_tail + TOKQ_SIZE) % TOKQ_SIZE;
    pthread_mutex_unlock(&tokq_lock);
    return n;
}

/* Append one UTF-8 encoded code point */
static int put_utf8(char *o, unsigned cp)
{
    if (cp < 0x80) { o[0] = cp; return 1; }
    if (cp < 0x800) { o[0] = 0xC0 | (cp >> 6); o[1] = 0x80 | (cp & 0x3F); return 2; }
    o[0] = 0xE0 | (cp >> 12); o[1] = 0x80 | ((cp >> 6) & 0x3F); o[2] = 0x80 | (cp & 0x3F);
    return 3;
}

/* Decode a JSON string body starting after the opening quote; whitespace escapes become spaces */
static const char *json_string(const char *p, const char *end, char *out, int cap)
{
    int n = 0;
    while (p < end && *p != '"') {
        char ch = *p++;
        if (ch == '\\' && p < end) {
            char e = *p++;
            if (e == 'u' && p + 4 <= end) {
                char hex[5] = { p[0], p[1], p[2], p[3], 0 };
                unsigned cp = (unsigned)strtoul(hex, NULL, 16);
                p += 4;
                if (cp >= 0xD800 && cp <= 0xDFFF)
                    cp = '?';                   /* surrogate pairs (emoji): not worth rendering */
                if (n + 3 < cap)
                    n += put_utf8(out + n, cp);
                continue;
            }
            ch = e == 'n' || e == 't' || e == 'r' ? ' ' : e;
        }
        if (n + 1 < cap)
            out[n++] = ch;
    }
    out[n] = 0;
    return p;
}

/* Split a text piece into words and queue them */
static void queue_words(char *s, int src)
{
    int queued = 0;
    for (char *w = strtok(s, " "); w && queued < 64; w = strtok(NULL, " ")) {
        if (!*w)
            continue;
        if (strlen(w) > TOK_MAX_BYTES) {
            int cut = TOK_MAX_BYTES;
            while (cut > 0 && ((unsigned char)w[cut] & 0xC0) == 0x80)
                cut--;                          /* don't split a UTF-8 sequence */
            strcpy(w + cut, "\xE2\x80\xA6");
        }
        tokq_push(w, src);
        queued++;
    }
}

static void scan_payload(const char *p, const char *end, int src)
{
    static const char *keys[] = { "\"content\":\"", "\"reasoning_content\":\"", "\"reasoning\":\"", "\"text\":\"" };
    char buf[4096];

    while (p < end) {
        const char *best = NULL;
        size_t best_len = 0;
        for (int k = 0; k < 4; k++) {
            const char *hit = memmem(p, end - p, keys[k], strlen(keys[k]));
            if (hit && (!best || hit < best)) {
                best = hit;
                best_len = strlen(keys[k]);
            }
        }
        if (!best)
            return;
        p = json_string(best + best_len, end, buf, sizeof(buf));
        queue_words(buf, src);
    }
}

static void on_packet(u_char *user, const struct pcap_pkthdr *h, const u_char *pkt)
{
    int linktype = *(int *)user;
    const u_char *ip = pkt + (linktype == DLT_EN10MB ? 14 : linktype == DLT_NULL ? 4 : 0);
    const u_char *end = pkt + h->caplen;

    if (ip + 20 > end)
        return;
    int version = ip[0] >> 4, ihl, proto;
    const u_char *tcp;
    if (version == 4) {
        ihl = (ip[0] & 0x0F) * 4;
        proto = ip[9];
        tcp = ip + ihl;
    } else if (version == 6) {
        proto = ip[6];
        tcp = ip + 40;
    } else {
        return;
    }
    if (proto != 6 || tcp + 20 > end)
        return;
    int sport = (tcp[0] << 8) | tcp[1];
    const u_char *payload = tcp + ((tcp[12] >> 4) * 4);
    if (payload >= end)
        return;
    for (int i = 0; i < N_PORTS; i++)
        if (sport == vllm_ports[i])
            scan_payload((const char *)payload, (const char *)end, i);
}

static void *capture_thread(void *arg)
{
    (void)arg;
    char err[PCAP_ERRBUF_SIZE], filter[256];
    struct bpf_program prog;
    int linktype;

    snprintf(filter, sizeof(filter), "tcp and (src port %d or src port %d)", vllm_ports[0], vllm_ports[1]);
    while (!stop) {
        pcap_t *pc = pcap_open_live("lo", 65535, 0, 50, err);
        if (!pc) {
            sleep(5);
            continue;
        }
        linktype = pcap_datalink(pc);
        if (pcap_compile(pc, &prog, filter, 1, PCAP_NETMASK_UNKNOWN) != 0 || pcap_setfilter(pc, &prog) != 0) {
            pcap_close(pc);
            sleep(5);
            continue;
        }
        pcap_freecode(&prog);
        while (!stop && pcap_dispatch(pc, -1, on_packet, (u_char *)&linktype) >= 0)
            ;
        pcap_close(pc);
    }
    return NULL;
}

/* ---------------------------------------------------------------- simulation */

#define HORIZON_R       70.0        /* event horizon radius (px) */
#define SPAWN_R         232.0
#define MAX_WORDS       260
#define MAX_DUST        1200
#define MAX_WORDS_PER_S 14.0        /* legible words per second; the rest of the flow is dust */
#define N_STARS         160

typedef struct {
    double r, theta, vr;
    double width;           /* text advance at base size, measured once */
    double spin;            /* small individual tumble */
    int    src, alive;
    char   text[TOK_MAX_BYTES + 4];
} word;

typedef struct {
    double r, theta, vr, px, py, size;
    int    src, alive;
} dust;

static word   words[MAX_WORDS];
static dust   dusts[MAX_DUST];
static double star_r[N_STARS], star_a[N_STARS], star_b[N_STARS];
static double word_budget, dust_acc[3];

static double frand(void) { return rand() / (double)RAND_MAX; }

static void infall(double *r, double *theta, double *vr, double dt)
{
    double omega = 1.0 * pow(SPAWN_R / *r, 1.5);
    *vr -= 2.0e6 / (*r * *r) * dt;
    *r += *vr * dt;
    *theta += omega * dt;
}

static void spawn_word(cairo_t *measure, const tok_item *t)
{
    for (int i = 0; i < MAX_WORDS; i++) {
        word *w = &words[i];
        if (w->alive)
            continue;
        cairo_text_extents_t ext;
        memcpy(w->text, t->text, sizeof(w->text));
        cairo_text_extents(measure, w->text, &ext);
        w->width = ext.x_advance;
        w->src = t->src;
        w->r = SPAWN_R - frand() * 10;
        w->theta = (t->src == 0 ? M_PI : 0) + (frand() - 0.5) * 2.2;
        w->vr = -(30 + frand() * 25);
        w->spin = (frand() - 0.5) * 0.4;
        w->alive = 1;
        return;
    }
}

static void spawn_dust(int src)
{
    for (int i = 0; i < MAX_DUST; i++) {
        dust *d = &dusts[i];
        if (d->alive)
            continue;
        d->src = src;
        d->r = SPAWN_R - frand() * 20;
        d->theta = src < 2 ? (src == 0 ? M_PI : 0) + (frand() - 0.5) * 2.6 : frand() * 2 * M_PI;
        d->vr = -(40 + frand() * 40);
        d->size = 0.8 + frand() * 1.0;
        d->px = SIZE / 2.0 + d->r * cos(d->theta);
        d->py = SIZE / 2.0 + d->r * sin(d->theta);
        d->alive = 1;
        return;
    }
}

static void simulate(cairo_t *measure, const stats *s, double dt)
{
    tok_item t;
    int words_spawned = 0;

    /* Words: take captured tokens at a legible rate; skip the backlog so words stay live */
    word_budget = fmin(word_budget + MAX_WORDS_PER_S * dt, 3);
    while (tokq_len() > 40 && tokq_pop(&t))
        ;
    while (word_budget >= 1 && tokq_pop(&t)) {
        spawn_word(measure, &t);
        word_budget -= 1;
        words_spawned++;
    }

    /* Dust: the rest of the real token flow, per server; a little ambient drift when idle */
    double rates[3] = { s->tok_port[0], s->tok_port[1], s->tok_s < 1 ? 5 : 0 };
    for (int k = 0; k < 3; k++) {
        dust_acc[k] += fmin(rates[k] / 2, 200) * dt;     /* one streak per two tokens, capped */
        while (dust_acc[k] >= 1) {
            spawn_dust(k);
            dust_acc[k] -= 1;
        }
    }
    (void)words_spawned;

    for (int i = 0; i < MAX_WORDS; i++) {
        word *w = &words[i];
        if (!w->alive)
            continue;
        infall(&w->r, &w->theta, &w->vr, dt * 0.7);     /* words fall slower so they can be read */
        if (w->r <= HORIZON_R * 0.92)
            w->alive = 0;
    }
    for (int i = 0; i < MAX_DUST; i++) {
        dust *d = &dusts[i];
        if (!d->alive)
            continue;
        infall(&d->r, &d->theta, &d->vr, dt);
        if (d->r <= HORIZON_R)
            d->alive = 0;
    }
    for (int i = 0; i < N_STARS; i++)
        star_a[i] += dt * 0.01;
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

static rgb src_base(int src)
{
    static const rgb AMBIENT = { 0.55, 0.60, 0.75 };
    return src == 0 ? BLUE : src == 1 ? ORANGE : AMBIENT;
}

/* Far out: server color. Falling in: white-hot. At the horizon: redshifted and dim. */
static rgb infall_color(int src, double r, double *alpha)
{
    static const rgb HOT = { 1.0, 0.96, 0.88 };
    static const rgb REDSHIFT = { 0.85, 0.12, 0.08 };
    double t = clamp01((SPAWN_R - r) / (SPAWN_R - HORIZON_R));
    double fade_in = clamp01((SPAWN_R - r) / 30);
    rgb c;
    if (t < 0.65) {
        c = lerp(src_base(src), HOT, t / 0.65 * 0.6);
        *alpha = fade_in;
    } else if (t < 0.88) {
        c = lerp(lerp(src_base(src), HOT, 0.6), HOT, (t - 0.65) / 0.23);
        *alpha = 1;
    } else {
        double k = (t - 0.88) / 0.12;
        c = lerp(HOT, REDSHIFT, clamp01(k * 1.4));
        *alpha = 1 - clamp01(k);
    }
    return c;
}

static void init_stars(void)
{
    for (int i = 0; i < N_STARS; i++) {
        star_r[i] = 40 + sqrt(frand()) * 210;
        star_a[i] = frand() * 2 * M_PI;
        star_b[i] = 0.25 + frand() * 0.75;
    }
}

/* Static bottom shade (cached overlay) and a cached text layer that is redrawn only
 * when something on it changes; numbers update 4x a second. Idle drops to FPS_IDLE. */
#define FPS_IDLE        15
static cairo_surface_t *fg_cache, *hud_cache;
static char hud_key[256];

static void build_caches(void)
{
    const double c = SIZE / 2.0;
    fg_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, SIZE);
    cairo_t *cr = cairo_create(fg_cache);
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
    cairo_destroy(cr);
    hud_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, SIZE);
}

static void update_hud(const stats *s, const shown_t *sh, double t, rgb core)
{
    const double c = SIZE / 2.0;
    static double tok, watts, next;
    char key[256], txt[64];

    if (t >= next || t < next - 1) {
        next = t + 0.25;
        tok = shown_rate(sh->tok);
        watts = s->power[0] + s->power[1];
    }
    snprintf(key, sizeof(key), "%d|%.0f|%.0f|%d|%d|%d|%.1f%.1f%.1f", s->tok_s < 0.5 && s->running == 0, tok, watts,
             s->temp[0], s->temp[1], s->running, core.r, core.g, core.b);
    if (!strcmp(key, hud_key))
        return;
    strcpy(hud_key, key);

    cairo_t *cr = cairo_create(hud_cache);
    cairo_set_operator(cr, CAIRO_OPERATOR_CLEAR);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    if (s->tok_s < 0.5 && s->running == 0) {
        text_center(cr, c, c, 24, 1, DIM, "IDLE");
    } else {
        snprintf(txt, sizeof(txt), "%.0f", tok);
        text_center(cr, c, c - 9, tok >= 1000 ? 40 : 50, 1, WHITE, txt);
        text_center(cr, c, c + 28, 15, 1, core, rate_unit_uc());
    }
    snprintf(txt, sizeof(txt), "%.0f W", watts);
    text_center(cr, c, c + 138, 34, 1, WHITE, txt);
    snprintf(txt, sizeof(txt), "%d\xC2\xB0", s->temp[0]);
    text_center(cr, c - 72, c + 180, 26, 1, BLUE, txt);
    snprintf(txt, sizeof(txt), "%d\xC2\xB0", s->temp[1]);
    text_center(cr, c + 72, c + 180, 26, 1, ORANGE, txt);
    if (s->running && !gpu_source) {
        snprintf(txt, sizeof(txt), "%d req", s->running);
        text_center(cr, c, c + 180, 18, 1, DIM, txt);
    }
    cairo_destroy(cr);
}

static void render(cairo_t *trail_cr, cairo_surface_t *trail, cairo_t *cr,
                   const stats *s, const shown_t *sh, double t)
{
    const double c = SIZE / 2.0;
    double total_w = s->power[0] + s->power[1];
    double heat = clamp01(total_w / POWER_MAX);
    rgb core = heat_color(heat);

    /* Dust trails live on their own surface, faded each frame */
    cairo_set_operator(trail_cr, CAIRO_OPERATOR_OVER);
    cairo_set_source_rgba(trail_cr, BG.r, BG.g, BG.b, 0.2);
    cairo_paint(trail_cr);

    /* Lensed starfield: apparent radius from the point-lens equation, so stars bunch into an Einstein ring */
    cairo_set_operator(trail_cr, CAIRO_OPERATOR_ADD);
    {
        const double theta_e = HORIZON_R * 1.35;
        for (int i = 0; i < N_STARS; i++) {
            double b = star_r[i];
            double r_app = (b + sqrt(b * b + 4 * theta_e * theta_e)) / 2;
            if (r_app > SIZE / 2.0 - 8)
                continue;
            double tw = star_b[i] * (0.7 + 0.3 * sin(t * (1 + i % 5) + i));
            double x = c + r_app * cos(star_a[i]), y = c + r_app * sin(star_a[i]);
            cairo_set_source_rgba(trail_cr, 0.8, 0.85, 1.0, 0.08 * tw);
            cairo_rectangle(trail_cr, x, y, 1.3, 1.3);
            cairo_fill(trail_cr);
        }
    }

    cairo_set_line_cap(trail_cr, CAIRO_LINE_CAP_ROUND);
    for (int i = 0; i < MAX_DUST; i++) {
        dust *d = &dusts[i];
        if (!d->alive)
            continue;
        double a;
        rgb col = infall_color(d->src, d->r, &a);
        double x = c + d->r * cos(d->theta), y = c + d->r * sin(d->theta);
        cairo_set_source_rgba(trail_cr, col.r, col.g, col.b, 0.55 * a);
        cairo_set_line_width(trail_cr, d->size);
        cairo_move_to(trail_cr, d->px, d->py);
        cairo_line_to(trail_cr, x, y);
        cairo_stroke(trail_cr);
        d->px = x;
        d->py = y;
    }
    cairo_set_operator(trail_cr, CAIRO_OPERATOR_OVER);

    cairo_set_source_surface(cr, trail, 0, 0);
    cairo_paint(cr);

    /* Accretion glow */
    {
        double a = 0.28 + 0.45 * heat;
        cairo_pattern_t *g = cairo_pattern_create_radial(c, c, HORIZON_R * 0.9, c, c, HORIZON_R * 2.7);
        cairo_pattern_add_color_stop_rgba(g, 0.0, core.r, core.g, core.b, a);
        cairo_pattern_add_color_stop_rgba(g, 0.22, core.r, core.g, core.b, a * 0.35);
        cairo_pattern_add_color_stop_rgba(g, 1.0, core.r, core.g, core.b, 0);
        cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
        cairo_set_source(cr, g);
        cairo_arc(cr, c, c, HORIZON_R * 2.7, 0, 2 * M_PI);     /* only the glow's own area */
        cairo_fill(cr);
        cairo_pattern_destroy(g);
    }

    /* Words: tangent to their orbit far out, turning to point at the hole and stretching as they fall */
    cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, 17);
    for (int i = 0; i < MAX_WORDS; i++) {
        const word *w = &words[i];
        if (!w->alive)
            continue;
        double a;
        rgb col = infall_color(w->src, w->r, &a);
        double k = clamp01((SPAWN_R - w->r) / (SPAWN_R - HORIZON_R));
        double tidal = k * k;
        double tangent = w->theta + M_PI / 2;
        double inward = w->theta + M_PI;
        double ang = tangent + (inward - tangent) * clamp01((k - 0.35) / 0.5) + w->spin * (1 - k);
        double sx = (1.0 - 0.25 * k) * (1 + 1.6 * tidal);        /* stretch along the fall line */
        double sy = (1.0 - 0.25 * k) / (1 + 2.2 * tidal);        /* and squeeze across it */
        double x = c + w->r * cos(w->theta), y = c + w->r * sin(w->theta);
        cairo_save(cr);
        cairo_translate(cr, x, y);
        cairo_rotate(cr, ang);
        cairo_scale(cr, sx, sy);
        cairo_move_to(cr, -w->width / 2, 6);
        cairo_set_source_rgba(cr, col.r, col.g, col.b, a * 0.95);
        cairo_show_text(cr, w->text);
        cairo_restore(cr);
    }
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);

    /* Photon ring plus a faint secondary image ring */
    {
        rgb ring = lerp(core, (rgb){ 1, 1, 1 }, 0.4);
        cairo_set_line_width(cr, 3.0);
        cairo_new_sub_path(cr);
        cairo_arc(cr, c, c, HORIZON_R + 1.5, 0, 2 * M_PI);
        cairo_set_source_rgba(cr, ring.r, ring.g, ring.b, 0.8 + 0.2 * sin(t * 3.1));
        cairo_stroke(cr);
        cairo_set_line_width(cr, 1.0);
        cairo_new_sub_path(cr);
        cairo_arc(cr, c, c, HORIZON_R + 7, 0, 2 * M_PI);
        cairo_set_source_rgba(cr, ring.r, ring.g, ring.b, 0.25);
        cairo_stroke(cr);
    }
    cairo_new_sub_path(cr);
    cairo_arc(cr, c, c, HORIZON_R, 0, 2 * M_PI);
    cairo_set_source_rgb(cr, 0, 0, 0);
    cairo_fill(cr);


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

    cairo_set_source_surface(cr, fg_cache, 0, 0);
    cairo_paint(cr);

    update_hud(s, sh, t, core);
    cairo_set_source_surface(cr, hud_cache, 0, 0);
    cairo_paint(cr);
}

/* ---------------------------------------------------------------- main */

static const char *demo_words[] = {
    "The", "model", "attends", "to", "every", "token", "in", "context,", "and", "each", "layer",
    "refines", "the", "residual", "stream.", "def", "forward(self,", "x):", "return", "softmax(",
    "logits)", "Therefore,", "the", "answer", "is", "42.", "HMAC-SHA256", "signature", "fixtures",
    "reliability", "### Option 1:", "tokens/sec", "latency", "throughput", "KV", "cache",
};

int main(int argc, char **argv)
{
    cairo_surface_t *surf, *trail;
    cairo_t *cr, *trail_cr;
    tjhandle tj;
    unsigned char *jpeg = NULL;
    stats s = { 0 };
    shown_t sh = { 0 };
    double last, next_poll = 0, t0, demo_acc = 0;
    int fd = -1, bench = 0;
    pthread_t cap;

    const char *source = getenv("LLM_REACTOR_SOURCE");
    gpu_source = source && !strcasecmp(source, "gpu");
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gpu-load"))
            gpu_source = 1;
        else if (!strcmp(argv[i], "--demo"))
            demo = 1;
        else if (!strcmp(argv[i], "--bench"))
            bench = 1;
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);
    srand((unsigned)time(NULL));
    init_stars();

    build_caches();
    surf = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cr = cairo_create(surf);
    trail = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    trail_cr = cairo_create(trail);
    cairo_set_source_rgb(trail_cr, BG.r, BG.g, BG.b);
    cairo_paint(trail_cr);
    /* Measuring context uses the same font as the words */
    cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, 17);
    tj = tj3Init(TJINIT_COMPRESS);
    tj3Set(tj, TJPARAM_QUALITY, JPEG_QUALITY);
    tj3Set(tj, TJPARAM_SUBSAMP, TJSAMP_420);

    if (bench || demo) {
        /* Feed words from a fixed list so the scene can be seen without live traffic */
    } else {
        pthread_create(&cap, NULL, capture_thread, NULL);
        pthread_detach(cap);
    }

    if (bench) {
        s.power[0] = 440; s.power[1] = 430; s.temp[0] = 54; s.temp[1] = 71;
        s.tok_port[0] = 800; s.tok_port[1] = 800; s.tok_s = 1600; s.running = 8;
        sh.zotac = 0.95; sh.tuf = 0.9; sh.tok = 1600;
        for (int i = 0; i < FPS * 12; i++) {
            for (int k = 0; k < 3; k++)
                tokq_push(demo_words[rand() % (sizeof(demo_words) / sizeof(*demo_words))], rand() % 2);
            simulate(cr, &s, 1.0 / FPS);
            render(trail_cr, trail, cr, &s, &sh, i / (double)FPS);
            cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
            cairo_set_font_size(cr, 17);
        }
        size_t len = 0;
        int nw = 0, nd = 0;
        double b0 = now_s();
        for (int i = 0; i < 200; i++) {
            for (int k = 0; k < 3; k++)
                tokq_push(demo_words[rand() % (sizeof(demo_words) / sizeof(*demo_words))], rand() % 2);
            simulate(cr, &s, 1.0 / FPS);
            render(trail_cr, trail, cr, &s, &sh, i / (double)FPS);
            cairo_surface_flush(surf);
            tj3Compress8(tj, cairo_image_surface_get_data(surf), SIZE, cairo_image_surface_get_stride(surf),
                         SIZE, TJPF_BGRX, &jpeg, &len);
            cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
            cairo_set_font_size(cr, 17);
        }
        for (int i = 0; i < MAX_WORDS; i++) nw += words[i].alive;
        for (int i = 0; i < MAX_DUST; i++) nd += dusts[i].alive;
        printf("%.2f ms/frame, jpeg %zu bytes, %d words, %d dust\n", (now_s() - b0) * 1000 / 200, len, nw, nd);
        cairo_surface_write_to_png(surf, "horizon_preview.png");
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
                if (gpu_source)
                    gpu_rate_poll(&s);
                else
                    vllm_poll(&s, t);
            }
        }
        if (demo) {
            demo_acc += s.tok_s / 8 * dt;
            while (demo_acc >= 1) {
                tokq_push(demo_words[rand() % (sizeof(demo_words) / sizeof(*demo_words))], rand() % 2);
                demo_acc -= 1;
            }
        }

        k = dt * 4 > 1 ? 1 : dt * 4;
        sh.zotac += (s.load[0] - sh.zotac) * k;
        sh.tuf   += (s.load[1] - sh.tuf) * k;
        sh.tok   += (s.tok_s - sh.tok) * k;

        cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
        cairo_set_font_size(cr, 17);
        simulate(cr, &s, dt);
        render(trail_cr, trail, cr, &s, &sh, t - t0);
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
        double spare = 1.0 / (idle_now ? FPS_IDLE : FPS) - (now_s() - t);
        if (spare > 0) {
            struct timespec ts = { 0, (long)(spare * 1e9) };
            nanosleep(&ts, NULL);
        }
    }

    tj3Free(jpeg);
    tj3Destroy(tj);
    cairo_destroy(trail_cr);
    cairo_surface_destroy(trail);
    cairo_destroy(cr);
    cairo_surface_destroy(surf);
    if (nvml_ok)
        nvmlShutdown();
    if (fd >= 0)
        close(fd);
    return 0;
}
