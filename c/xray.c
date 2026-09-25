/*
 * xray: a glowing holographic body scan of an android on the iCUE LINK AIO pump LCD,
 * driven by the machine's own vital signs.
 *
 * The CPU is the heart: it beats faster and brighter with total CPU load, and a ring
 * of 32 ticks around it pulses with each hardware thread. RAM fills the lungs. NVMe
 * reads churn the stomach and writes flow down the gut. Network traffic is blood in
 * the veins (download flowing in to the heart, upload pumped out). The two GPUs are
 * the two hemispheres of the brain, and PCIe traffic to them pulses up and down the
 * spinal cord. Page faults twitch the nerves along the ribs, interrupts put blips on
 * the ECG, and pressure stall (PSI) makes the breathing laboured. CPU temperature
 * (Tctl) shifts the whole scan from cool cyan to feverish red. The rim shows the
 * patient's age (uptime) and CPU package watts. A scan line sweeps down the body.
 * The skeleton is an image in assets/xray/ (made with an image model); everything
 * else is drawn with cairo. Most of the scan is one static alpha mask that is tinted
 * on the fly, so the fever colour costs nothing extra.
 * Run with --demo to simulate data, --showcase for a scripted 36 s arc, --bench to
 * write preview PNGs.
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

__attribute__((unused)) static void set_rgb(cairo_t *cr, rgb c) { cairo_set_source_rgb(cr, c.r, c.g, c.b); }

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


/* ---- system sensors ---- */

/*
 * Machine vital signs from /proc and /sys/class/hwmon. hwmon devices are found by
 * name at startup (never by number). Rates are differenced between polls, so poll
 * at most a couple of times a second. Everything is parsed from one read() per file.
 */
#define MAX_THREADS     64
#define N_NVME          3
#define NET_IF          "enp12s0"
#define NET_IF2         "tailscale0"

typedef struct {
    int    n_threads;
    double thread[MAX_THREADS];     /* per hardware thread busy, 0..1 */
    double cpu;                     /* all threads, 0..1 */
    double tctl;                    /* k10temp Tctl, C */
    double mb_cpu, mb_temp, vrm;    /* asusec, C */
    int    fan_rpm;                 /* asusec fan1 */
    double ram_total, ram_used, ram_cached;     /* GiB; used = total - available */
    double dimm_temp[2];            /* spd5118, C */
    double nvme_temp[N_NVME];       /* C */
    double disk_rd, disk_wr;        /* all NVMe, bytes/s */
    double net_rx, net_tx;          /* NET_IF, bytes/s */
    double ts_rx, ts_tx;            /* NET_IF2 (rides on NET_IF, so not added to it) */
    double intr_s, ctxt_s, forks_s; /* /proc/stat: interrupts, context switches, forks per second */
    int    procs_running;
    double pgfault_s, pgmajfault_s; /* /proc/vmstat, per second */
    double psi_cpu, psi_mem, psi_io;    /* /proc/pressure "some avg10", % of time stalled */
    double load1, uptime;           /* /proc/loadavg, /proc/uptime (s) */
    double pkg_w;                   /* CPU package watts from RAPL; -1 if unreadable (needs root) */
    double pcie_rx[N_GPUS], pcie_tx[N_GPUS];    /* NVML, bytes/s from each GPU's view: rx = host to GPU */
} sys_stats;

static char hw_tctl[300], hw_mb_cpu[300], hw_mb[300], hw_vrm[300], hw_fan[300];
static char hw_dimm[2][300], hw_nvme[N_NVME][300];

/* Read a small file into buf; returns length or -1 */
static int read_small(const char *path, char *buf, int cap)
{
    int fd = open(path, O_RDONLY | O_CLOEXEC), got = 0;
    if (fd < 0)
        return -1;
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

static int read_milli(const char *path, double *out)      /* millidegrees -> C */
{
    char b[32];
    if (!*path || read_small(path, b, sizeof(b)) <= 0)
        return 0;
    *out = strtol(b, NULL, 10) / 1000.0;
    return 1;
}

static void sys_init(void)
{
    DIR *d = opendir("/sys/class/hwmon");
    struct dirent *e;
    int n_dimm = 0, n_nvme = 0;
    char path[300], name[64], label[64];

    if (!d)
        return;
    while ((e = readdir(d))) {
        if (strncmp(e->d_name, "hwmon", 5) != 0)
            continue;
        snprintf(path, sizeof(path), "/sys/class/hwmon/%s/name", e->d_name);
        if (read_small(path, name, sizeof(name)) <= 0)
            continue;
        name[strcspn(name, "\n")] = 0;
        if (!strcmp(name, "k10temp")) {
            snprintf(hw_tctl, sizeof(hw_tctl), "/sys/class/hwmon/%s/temp1_input", e->d_name);
        } else if (!strcmp(name, "asusec")) {
            snprintf(hw_fan, sizeof(hw_fan), "/sys/class/hwmon/%s/fan1_input", e->d_name);
            for (int i = 1; i <= 8; i++) {
                snprintf(path, sizeof(path), "/sys/class/hwmon/%s/temp%d_label", e->d_name, i);
                if (read_small(path, label, sizeof(label)) <= 0)
                    continue;
                label[strcspn(label, "\n")] = 0;
                char *dst = !strcmp(label, "CPU") ? hw_mb_cpu : !strcmp(label, "Motherboard") ? hw_mb
                          : !strcmp(label, "VRM") ? hw_vrm : NULL;
                if (dst)
                    snprintf(dst, 300, "/sys/class/hwmon/%s/temp%d_input", e->d_name, i);
            }
        } else if (!strcmp(name, "spd5118") && n_dimm < 2) {
            snprintf(hw_dimm[n_dimm++], 300, "/sys/class/hwmon/%s/temp1_input", e->d_name);
        } else if (!strcmp(name, "nvme") && n_nvme < N_NVME) {
            snprintf(hw_nvme[n_nvme++], 300, "/sys/class/hwmon/%s/temp1_input", e->d_name);
        }
    }
    closedir(d);
}

static double psi_some10(const char *path)
{
    char b[256];
    const char *p;
    if (read_small(path, b, sizeof(b)) <= 0 || !(p = strstr(b, "avg10=")))
        return 0;
    return strtod(p + 6, NULL);
}

/* Value after "key:" in a /proc/meminfo-style buffer */
static double kv_num(const char *buf, const char *key)
{
    const char *p = strstr(buf, key);
    return p ? strtod(p + strlen(key), NULL) : 0;
}

static void sys_poll(sys_stats *s, double t)
{
    static char buf[1 << 16];
    static unsigned long long last_busy[MAX_THREADS], last_all[MAX_THREADS];
    static double last_rd, last_wr, last_rx, last_tx, last_trx, last_ttx, last_t, last_net_t;
    static double last_intr, last_ctxt, last_forks, last_pgf, last_pgmaj, last_energy;
    static int have, rapl_ok = 1, pcie_k;
    double dt = t - last_t, intr = 0, ctxt = 0, forks = 0, pgf = 0, pgmaj = 0, energy = 0;

    /* CPU: per-thread busy time, differenced */
    if (read_small("/proc/stat", buf, sizeof(buf)) > 0) {
        char *p = buf;
        int n = 0;
        double sum = 0;
        while ((p = strstr(p, "\ncpu")) && n < MAX_THREADS) {
            p += 4;
            if (*p < '0' || *p > '9')
                continue;
            int id = (int)strtol(p, &p, 10);
            unsigned long long v[8] = { 0 }, all = 0;
            for (int k = 0; k < 8; k++) {
                v[k] = strtoull(p, &p, 10);
                all += v[k];
            }
            unsigned long long busy = all - v[3] - v[4];        /* minus idle, iowait */
            if (id >= 0 && id < MAX_THREADS) {
                if (have && all > last_all[id])
                    s->thread[id] = clamp01((double)(busy - last_busy[id]) / (all - last_all[id]));
                last_busy[id] = busy;
                last_all[id] = all;
                sum += s->thread[id];
                if (id + 1 > n)
                    n = id + 1;
            }
        }
        s->n_threads = n;
        s->cpu = n ? sum / n : 0;
        s->procs_running = (int)kv_num(buf, "\nprocs_running");
        intr = kv_num(buf, "\nintr");
        ctxt = kv_num(buf, "\nctxt");
        forks = kv_num(buf, "\nprocesses");
    }
    if (read_small("/proc/vmstat", buf, sizeof(buf)) > 0) {
        pgf = kv_num(buf, "\npgfault");
        pgmaj = kv_num(buf, "\npgmajfault");
    }
    s->psi_cpu = psi_some10("/proc/pressure/cpu");
    s->psi_mem = psi_some10("/proc/pressure/memory");
    s->psi_io  = psi_some10("/proc/pressure/io");
    if (read_small("/proc/loadavg", buf, 64) > 0)
        s->load1 = strtod(buf, NULL);
    if (read_small("/proc/uptime", buf, 64) > 0)
        s->uptime = strtod(buf, NULL);
    if (rapl_ok && read_small("/sys/class/powercap/intel-rapl:0/energy_uj", buf, 64) > 0) {
        energy = strtod(buf, NULL);
    } else {
        rapl_ok = 0;                /* root only: fall back quietly */
        s->pkg_w = -1;
    }

    /* PCIe throughput: NVML samples each counter for ~25 ms, so read one per poll */
    if (nvml_ok) {
        int gi = pcie_k / 2 % N_GPUS;
        unsigned int kb;
        if (nvml_dev[gi] && nvmlDeviceGetPcieThroughput(nvml_dev[gi], pcie_k % 2 ? NVML_PCIE_UTIL_TX_BYTES
                                                                   : NVML_PCIE_UTIL_RX_BYTES, &kb) == NVML_SUCCESS)
            (pcie_k % 2 ? s->pcie_tx : s->pcie_rx)[gi] = kb * 1024.0;
        pcie_k++;
    }

    read_milli(hw_tctl, &s->tctl);
    read_milli(hw_mb_cpu, &s->mb_cpu);
    read_milli(hw_mb, &s->mb_temp);
    read_milli(hw_vrm, &s->vrm);
    for (int i = 0; i < 2; i++)
        read_milli(hw_dimm[i], &s->dimm_temp[i]);
    for (int i = 0; i < N_NVME; i++)
        read_milli(hw_nvme[i], &s->nvme_temp[i]);
    if (*hw_fan && read_small(hw_fan, buf, 32) > 0)
        s->fan_rpm = (int)strtol(buf, NULL, 10);

    if (read_small("/proc/meminfo", buf, sizeof(buf)) > 0) {
        double total = kv_num(buf, "MemTotal:"), avail = kv_num(buf, "MemAvailable:");
        s->ram_total  = total / 1048576.0;
        s->ram_used   = (total - avail) / 1048576.0;
        s->ram_cached = kv_num(buf, "\nCached:") / 1048576.0;
    }

    /* NVMe throughput: sectors read (field 6) and written (field 10), whole disks only */
    double rd = 0, wr = 0, rx = 0, tx = 0, trx = 0, ttx = 0;
    if (read_small("/proc/diskstats", buf, sizeof(buf)) > 0) {
        for (char *line = buf; line && *line; line = strchr(line, '\n'), line = line ? line + 1 : NULL) {
            char dev[32];
            unsigned long long r, w;
            int nn;
            if (sscanf(line, "%*u %*u %31s %*u %*u %llu %*u %*u %*u %llu", dev, &r, &w) == 3 &&
                sscanf(dev, "nvme%dn1", &nn) == 1 && strlen(dev) <= 7) {
                rd += r * 512.0;
                wr += w * 512.0;
            }
        }
    }
    if (read_small("/proc/net/dev", buf, sizeof(buf)) > 0) {
        char *p;
        if ((p = strstr(buf, NET_IF ":")))
            sscanf(p + strlen(NET_IF ":"), "%lf %*f %*f %*f %*f %*f %*f %*f %lf", &rx, &tx);
        if ((p = strstr(buf, NET_IF2 ":")))
            sscanf(p + strlen(NET_IF2 ":"), "%lf %*f %*f %*f %*f %*f %*f %*f %lf", &trx, &ttx);
    }
    if (have && dt > 0.05) {
        s->intr_s = fmax(0, intr - last_intr) / dt;
        s->ctxt_s = fmax(0, ctxt - last_ctxt) / dt;
        s->forks_s = fmax(0, forks - last_forks) / dt;
        s->pgfault_s = fmax(0, pgf - last_pgf) / dt;
        s->pgmajfault_s = fmax(0, pgmaj - last_pgmaj) / dt;
        if (rapl_ok && energy >= last_energy)          /* skip the counter wrapping */
            s->pkg_w = (energy - last_energy) / 1e6 / dt;
        s->disk_rd = fmax(0, rd - last_rd) / dt;
        s->disk_wr = fmax(0, wr - last_wr) / dt;
    }
    /* NIC counters only tick about once a second: measure between changes, not polls */
    double ndt = t - last_net_t;
    if (have && ((rx != last_rx || tx != last_tx) || ndt > 2.5) && ndt > 0.05) {
        s->net_rx = fmax(0, rx - last_rx) / ndt;
        s->net_tx = fmax(0, tx - last_tx) / ndt;
        s->ts_rx  = fmax(0, trx - last_trx) / ndt;
        s->ts_tx  = fmax(0, ttx - last_ttx) / ndt;
        last_rx = rx, last_tx = tx, last_trx = trx, last_ttx = ttx, last_net_t = t;
    } else if (!have) {
        last_rx = rx, last_tx = tx, last_trx = trx, last_ttx = ttx, last_net_t = t;
    }
    last_rd = rd, last_wr = wr;
    last_intr = intr, last_ctxt = ctxt, last_forks = forks, last_pgf = pgf, last_pgmaj = pgmaj, last_energy = energy;
    last_t = t;
    have = 1;
}

/* ---- end system sensors ---- */

static double frand(void) { return rand() / (double)RAND_MAX; }

/* Simulated machine: every sensor wanders on its own period */
static void sys_demo(sys_stats *s, double t)
{
    double busy = pow(0.5 + 0.5 * sin(t * 0.19 - 1.2), 1.4);
    s->n_threads = 32;
    s->cpu = 0;
    for (int i = 0; i < 32; i++) {
        double own = 0.5 + 0.5 * sin(t * (0.5 + 0.09 * i) + i * 2.3);
        double v = busy * (0.35 + 0.9 * own) + 0.03 * frand();
        if (i % 7 == 3)                 /* a couple of threads always doing something */
            v = fmax(v, 0.25 + 0.2 * sin(t * 1.3 + i));
        s->thread[i] = clamp01(v);
        s->cpu += s->thread[i] / 32;
    }
    s->tctl = 42 + 52 * pow(0.5 + 0.5 * sin(t * 0.19 - 2.0), 1.3);
    s->ram_total = 91;
    s->ram_used = 14 + 62 * (0.5 + 0.5 * sin(t * 0.071 + 0.5));
    s->ram_cached = 20;
    s->disk_rd = 3.0e9 * pow(fmax(0, sin(t * 0.43)), 4) + 2e5;
    s->disk_wr = 1.6e9 * pow(fmax(0, sin(t * 0.31 + 2)), 6);
    s->net_rx = 2.2e8 * pow(0.5 + 0.5 * sin(t * 0.27 + 1), 3) + 3e4;
    s->net_tx = 6e7 * pow(fmax(0, sin(t * 0.23 + 4)), 2) + 1e4;
    s->mb_temp = 38, s->vrm = 45 + 20 * busy, s->fan_rpm = 0;
    s->intr_s = 25e3 + 600e3 * busy * (0.7 + 0.3 * sin(t * 1.7));
    s->ctxt_s = 40e3 + 900e3 * busy;
    s->pgfault_s = 8e3 + 1.5e6 * pow(fmax(0, sin(t * 0.29 + 1)), 5);
    s->pgmajfault_s = 60e3 * pow(fmax(0, sin(t * 0.41 + 3)), 12);
    s->psi_cpu = 30 * pow(busy, 3);
    s->psi_mem = 2 + 25 * pow(fmax(0, sin(t * 0.071 + 0.5)), 6);
    s->psi_io = 12 * pow(fmax(0, sin(t * 0.43)), 4);
    s->load1 = 0.5 + 30 * busy;
    s->uptime = 9 * 86400 + 5 * 3600 + t;
    s->pkg_w = 38 + 190 * busy;
    for (int i = 0; i < N_GPUS; i++) {
        s->pcie_rx[i] = 5e6 + 9e9 * pow(fmax(0, sin(t * 0.37 + i * 2)), 8);
        s->pcie_tx[i] = 3e6 + 3e9 * pow(fmax(0, sin(t * 0.33 + i * 3 + 1)), 8);
    }
}

/*
 * --showcase: a scripted 36 s arc for filming or GIFs: a sleeping machine wakes up,
 * pulls in data, the GPUs light up, everything maxes out into a fever, then it cools
 * back down to rest.
 */
#define SHOW_LEN 36.0

static double smoothstep(double a, double b, double x)
{
    x = clamp01((x - a) / (b - a));
    return x * x * (3 - 2 * x);
}

static void showcase_poll(stats *g, sys_stats *s, double t)
{
    double u = fmod(t, SHOW_LEN);
    double up = smoothstep(3, 17, u) * (1 - smoothstep(26, 33, u));         /* overall effort */
    double peak = smoothstep(16, 19, u) * (1 - smoothstep(25, 28, u));      /* everything maxed */
    double heat = smoothstep(6, 21, u) * (1 - smoothstep(27, 35, u));       /* lags the load */

    s->n_threads = 32;
    s->cpu = 0;
    for (int i = 0; i < 32; i++) {
        /* threads wake one after another, then all max out */
        double wake = smoothstep(3 + i * 0.3, 5 + i * 0.3, u) * (1 - smoothstep(26, 32 - (i % 5), u));
        double own = 0.5 + 0.5 * sin(u * (0.9 + 0.11 * i) + i * 1.7);
        double v = 0.02 + wake * (0.25 + 0.45 * own) * (0.4 + 0.6 * up) + peak * 0.95;
        s->thread[i] = clamp01(v + 0.02 * frand());
        s->cpu += s->thread[i] / 32;
    }
    s->tctl = 41 + 53 * heat + 1.5 * peak * sin(u * 3);
    s->ram_total = 91;
    s->ram_used = 12 + 72 * smoothstep(5, 22, u) * (1 - smoothstep(28, 34, u));
    s->ram_cached = 18;
    s->disk_rd = 2e5 + 3.5e9 * smoothstep(6, 7, u) * (1 - smoothstep(11, 13, u)) + 2.5e9 * peak;
    s->disk_wr = 1.8e9 * smoothstep(19, 20, u) * (1 - smoothstep(27, 29, u)) + 5e7 * up;
    s->net_rx = 2e4 + 2.6e8 * smoothstep(8, 10, u) * (1 - smoothstep(24, 29, u));
    s->net_tx = 1e4 + 1.2e8 * smoothstep(13, 15, u) * (1 - smoothstep(25, 30, u));
    s->mb_temp = 38, s->vrm = 45 + 20 * up, s->fan_rpm = 0;
    s->intr_s = 25e3 + 700e3 * up + 300e3 * peak;
    s->ctxt_s = 40e3 + 1e6 * up;
    s->pgfault_s = 6e3 + 1.8e6 * smoothstep(5, 7, u) * (1 - smoothstep(20, 23, u));
    s->pgmajfault_s = 50e3 * smoothstep(6, 7, u) * (1 - smoothstep(9, 10, u)) + 20 * up;
    s->psi_cpu = 35 * peak;
    s->psi_mem = 2 + 30 * smoothstep(18, 21, u) * (1 - smoothstep(26, 29, u));
    s->psi_io = 15 * smoothstep(6, 7, u) * (1 - smoothstep(11, 13, u));
    s->load1 = 0.4 + 34 * up;
    s->uptime = 12 * 86400 + 7 * 3600 + t;
    s->pkg_w = 36 + 110 * up + 90 * peak;
    /* both GPUs pull in their models over PCIe as they wake, then trade results */
    for (int i = 0; i < N_GPUS; i++) {
        double w = i ? 15 : 11, busy = smoothstep(w + 2, w + 3, u) * (1 - smoothstep(26, 29, u));
        s->pcie_rx[i] = 4e6 + 2.4e10 * smoothstep(w - 1, w, u) * (1 - smoothstep(w + 2, w + 3, u)) + 2e9 * busy;
        s->pcie_tx[i] = 2e6 + 1.5e9 * busy;
    }

    double g0 = smoothstep(11, 13, u) * (1 - smoothstep(27, 30, u));
    double g1 = smoothstep(15, 17, u) * (1 - smoothstep(26, 29, u));
    g->load[0] = clamp01(g0 * (0.85 + 0.15 * sin(u * 2.1)));
    g->load[1] = clamp01(g1 * (0.85 + 0.15 * sin(u * 1.7 + 1)));
    g->power[0] = 30 + 530 * g->load[0];
    g->power[1] = 34 + 520 * g->load[1];
    g->temp[0] = (int)(35 + 30 * g->load[0]);
    g->temp[1] = (int)(38 + 32 * g->load[1]);
    g->tok_port[0] = g->tok_port[1] = g->tok_s = 0;
    g->running = 0;
}

/* ---------------------------------------------------------------- scene */

/*
 * Layers, back to front, all added (CAIRO_OPERATOR_ADD) like light on film:
 *   bg        static RGB: deep blue-black, faint grid and grain
 *   fig       static A8 alpha mask of everything that only changes colour: bones (from
 *             the image), their glow, lung outlines and bronchi, vessels, gut, brain
 *             folds, the bezel. Painted every frame through the fever tint.
 *   core      static A8: the brightest bone, painted in a paler tint
 *   dynamic   lungs filling, heart and its thread ring, brain, stomach, blood, ECG,
 *             scan line, rim gauges
 *   hud       A8 text, redrawn at most 4 times a second when a number changes
 */
#define FPS_BUSY        20
#define FPS_IDLE        15
#define CX              240.0
#define CY              240.0

/* Organ positions, in screen pixels over the 480x480 skeleton image */
#define HEART_X         256.0
#define HEART_Y         258.0
#define LUNG_TOP        192.0
#define LUNG_BOT        318.0
#define STOM_X          268.0
#define STOM_Y          334.0
#define BRAIN_Y         80.0
#define ECG_Y           424.0
#define ECG_X0          128.0
#define ECG_X1          352.0
#define ECG_AMP         26.0
#define ECG_SPEED       110.0       /* px/s */
#define GAUGE_R         226.0
#define ARC_TEXT_R      208.0       /* baseline of the text bent along the top of the rim */

#define DISK_FULL       4.0e9       /* B/s that counts as flat out */
#define NET_FULL        3.0e8

typedef struct {
    double cpu, thread[MAX_THREADS], tctl, heat, ram, ram_gb, gpu[2], rd, wr, rx, tx;
    double intr, pf, pfmaj, psi, pcie_up, pcie_dn;      /* 0..1 activities */
    double pkg_w, load1, uptime;                        /* for the text */
} shown_t;

static cairo_surface_t *rim_mask, *brain_spr, *bg_surf, *fig_mask, *core_mask, *heart_spr, *lung_fill_mask, *hud_mask, *hud_shadow;
static cairo_path_t *lung_path[2], *stom_path;

/* Vessels and the gut: smooth curves resampled every VSTEP px */
#define VSTEP   2.0
#define VMAX    400
typedef struct { int n; float x[VMAX], y[VMAX]; } vpath;
enum { V_LARM, V_RARM, V_NECK, V_AORTA, N_VESSELS };
#define N_NERVES 10
static vpath vessels[N_VESSELS], gut, nerves[N_NERVES], spine;

/* nerve twitches (page faults) and ECG blips (interrupts) */
#define N_TWITCH 40
typedef struct { int nerve; double age, strong; } twitch;
static twitch twitches[N_TWITCH];
#define N_BLIPS 128
static double blip_t[N_BLIPS], blip_a[N_BLIPS];
static int    blip_n;
static double twitch_acc, blip_acc;

#define N_SPARKS 48
typedef struct { double x, y, age, life; int side; } spark;
static spark sparks[N_SPARKS];

#define N_BEATS 24
static double beat_t[N_BEATS];      /* recent beat times, for the ECG */
static int    beat_n;
static double beat_phase, breath_phase, churn, flow_rx, flow_tx, flow_gut, spark_acc[2];
static char   hud_key[96];
static double hud_next;

/* ---- colour */

/* The scan's colour with CPU temperature: cool cyan, lavender, rose, fever red */
static rgb scan_tint(double h)
{
    static const rgb st[] = {
        { 0.30, 0.86, 1.00 }, { 0.52, 0.74, 1.00 }, { 0.80, 0.56, 1.00 }, { 1.00, 0.42, 0.62 }, { 1.00, 0.26, 0.18 },
    };
    h = clamp01(h) * 4;
    int i = h >= 4 ? 3 : (int)h;
    return lerp(st[i], st[i + 1], h - i);
}

static rgb heart_tint(double h)
{
    static const rgb rose = { 1.00, 0.36, 0.52 }, fever = { 1.00, 0.30, 0.16 };
    return lerp(rose, fever, smoothstep(0.3, 1.0, h));
}

static rgb pale(rgb c, double k) { return lerp(c, (rgb){ 1, 1, 1 }, k); }

static void set_rgba(cairo_t *cr, rgb c, double a) { cairo_set_source_rgba(cr, c.r, c.g, c.b, a); }

/* ---- assets and static layers */

static cairo_surface_t *load_asset(const char *name)
{
    char exe[512], path[1024];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n > 0) {
        exe[n] = 0;
        char *slash = strrchr(exe, '/');
        if (slash)
            *slash = 0;
        snprintf(path, sizeof(path), "%s/assets/xray/%s", exe, name);
        cairo_surface_t *s = cairo_image_surface_create_from_png(path);
        if (cairo_surface_status(s) == CAIRO_STATUS_SUCCESS)
            return s;
        cairo_surface_destroy(s);
    }
    snprintf(path, sizeof(path), "assets/xray/%s", name);
    cairo_surface_t *s = cairo_image_surface_create_from_png(path);
    if (cairo_surface_status(s) != CAIRO_STATUS_SUCCESS) {
        fprintf(stderr, "xray: can't load %s (looked next to the binary and in ./assets/xray)\n", name);
        exit(1);
    }
    return s;
}

/* Three box blurs on an A8 buffer: close to a gaussian */
static void blur_a8(unsigned char *d, int w, int h, int stride, int r)
{
    int *tmp = malloc(sizeof(int) * (w > h ? w : h));
    for (int pass = 0; pass < 3; pass++) {
        for (int y = 0; y < h; y++) {
            unsigned char *row = d + y * stride;
            int acc = 0;
            for (int x = -r; x <= r; x++)
                acc += row[x < 0 ? 0 : x >= w ? w - 1 : x];
            for (int x = 0; x < w; x++) {
                tmp[x] = acc / (2 * r + 1);
                int a = x - r, b = x + r + 1;
                acc += row[b >= w ? w - 1 : b] - row[a < 0 ? 0 : a];
            }
            for (int x = 0; x < w; x++)
                row[x] = (unsigned char)tmp[x];
        }
        for (int x = 0; x < w; x++) {
            int acc = 0;
            for (int y = -r; y <= r; y++)
                acc += d[(y < 0 ? 0 : y >= h ? h - 1 : y) * stride + x];
            for (int y = 0; y < h; y++) {
                tmp[y] = acc / (2 * r + 1);
                int a = y - r, b = y + r + 1;
                acc += d[(b >= h ? h - 1 : b) * stride + x] - d[(a < 0 ? 0 : a) * stride + x];
            }
            for (int y = 0; y < h; y++)
                d[y * stride + x] = (unsigned char)tmp[y];
        }
    }
    free(tmp);
}

/* Catmull-Rom through pts, resampled every VSTEP px */
static void make_vpath(vpath *v, const double (*p)[2], int np)
{
    double px = p[0][0], py = p[0][1], carry = 0;
    v->n = 0;
    v->x[v->n] = px, v->y[v->n] = py, v->n++;
    for (int i = 0; i < np - 1; i++) {
        const double *p0 = p[i > 0 ? i - 1 : 0], *p1 = p[i], *p2 = p[i + 1], *p3 = p[i + 2 < np ? i + 2 : np - 1];
        for (int k = 1; k <= 60; k++) {
            double u = k / 60.0, u2 = u * u, u3 = u2 * u;
            double x = 0.5 * (2 * p1[0] + (-p0[0] + p2[0]) * u + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * u2 +
                              (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * u3);
            double y = 0.5 * (2 * p1[1] + (-p0[1] + p2[1]) * u + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * u2 +
                              (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * u3);
            double seg = hypot(x - px, y - py);
            while (carry + seg >= VSTEP && v->n < VMAX) {
                double f = (VSTEP - carry) / seg;
                px += (x - px) * f, py += (y - py) * f;
                seg = hypot(x - px, y - py);
                carry = 0;
                v->x[v->n] = px, v->y[v->n] = py, v->n++;
            }
            carry += seg;
            px = x, py = y;
        }
    }
}

/* Point at distance d along the path, pushed sideways by off px */
static void vpath_at(const vpath *v, double d, double off, double *x, double *y)
{
    double f = d / VSTEP;
    int i = (int)f;
    if (i >= v->n - 1)
        i = v->n - 2, f = i + 1;
    f -= i;
    double dx = v->x[i + 1] - v->x[i], dy = v->y[i + 1] - v->y[i], l = hypot(dx, dy) + 1e-9;
    *x = v->x[i] + dx * f - dy / l * off;
    *y = v->y[i] + dy * f + dx / l * off;
}

static void vpath_stroke(cairo_t *cr, const vpath *v, double off)
{
    double x, y;
    cairo_new_path(cr);
    for (int i = 0; i < v->n - 1; i++) {
        vpath_at(v, i * VSTEP, off, &x, &y);
        if (i == 0)
            cairo_move_to(cr, x, y);
        else
            cairo_line_to(cr, x, y);
    }
    cairo_stroke(cr);
}

/* Lungs, viewer's left then right (the right one has the notch for the heart) */
static void lung_shape(cairo_t *cr, int side)
{
    double m = side ? -1 : 1, c = CX;
    cairo_new_path(cr);
    cairo_move_to(cr, c - m * 22, LUNG_TOP);
    cairo_curve_to(cr, c - m * 40, LUNG_TOP - 4, c - m * 58, LUNG_TOP + 30, c - m * 62, LUNG_TOP + 70);
    cairo_curve_to(cr, c - m * 66, LUNG_BOT - 30, c - m * 66, LUNG_BOT - 4, c - m * 56, LUNG_BOT);
    cairo_curve_to(cr, c - m * 42, LUNG_BOT - 10, c - m * 26, LUNG_BOT - 14, c - m * 14, LUNG_BOT - 16);
    if (side) {             /* cardiac notch */
        cairo_curve_to(cr, c + 30, LUNG_BOT - 30, c + 34, LUNG_BOT - 54, c + 16, LUNG_BOT - 64);
        cairo_curve_to(cr, c + 12, LUNG_BOT - 80, c + 12, LUNG_TOP + 20, c + 22, LUNG_TOP);
    } else {
        cairo_curve_to(cr, c - 10, LUNG_BOT - 60, c - 12, LUNG_TOP + 30, c - 22, LUNG_TOP);
    }
    cairo_close_path(cr);
}

/* Bronchial tree: recursive branches with a little wobble */
static void bronchus(cairo_t *cr, double x, double y, double ang, double len, double w, int depth)
{
    if (depth == 0 || len < 3)
        return;
    double x2 = x + cos(ang) * len, y2 = y + sin(ang) * len;
    double bend = (frand() - 0.5) * 0.4;
    cairo_set_line_width(cr, w);
    cairo_move_to(cr, x, y);
    cairo_curve_to(cr, x + cos(ang + bend) * len * 0.5, y + sin(ang + bend) * len * 0.5, x2, y2, x2, y2);
    cairo_stroke(cr);
    double spread = 0.35 + frand() * 0.3;
    bronchus(cr, x2, y2, ang - spread, len * (0.68 + frand() * 0.12), w * 0.72, depth - 1);
    bronchus(cr, x2, y2, ang + spread, len * (0.68 + frand() * 0.12), w * 0.72, depth - 1);
}

static void stomach_shape(cairo_t *cr)
{
    double x = STOM_X, y = STOM_Y;
    cairo_new_path(cr);
    cairo_move_to(cr, x - 8, y - 22);
    cairo_curve_to(cr, x + 2, y - 34, x + 30, y - 30, x + 30, y - 8);
    cairo_curve_to(cr, x + 30, y + 14, x + 12, y + 24, x - 10, y + 22);
    cairo_curve_to(cr, x - 22, y + 21, x - 32, y + 16, x - 36, y + 8);
    cairo_curve_to(cr, x - 28, y + 6, x - 16, y + 7, x - 9, y + 2);
    cairo_curve_to(cr, x - 3, y - 4, x - 6, y - 12, x - 8, y - 22);
    cairo_close_path(cr);
}

/* Anatomical-ish heart centred on 0,0, apex to the lower right, great vessels on top */
static void heart_body(cairo_t *cr)
{
    cairo_new_path(cr);
    cairo_move_to(cr, -20, -14);
    cairo_curve_to(cr, -34, -8, -32, 14, -16, 24);
    cairo_curve_to(cr, -4, 32, 14, 36, 28, 30);
    cairo_curve_to(cr, 36, 22, 34, 4, 26, -8);
    cairo_curve_to(cr, 20, -18, 6, -22, -4, -20);
    cairo_curve_to(cr, -12, -20, -16, -18, -20, -14);
    cairo_close_path(cr);
}

#define BRAIN_X0  (CX - 40)       /* brain sprite placement */
#define BRAIN_Y0  (BRAIN_Y - 45)
#define LF_X0     (CX - 74)       /* full-lung sprite placement */
#define LF_Y0     (LUNG_TOP - 8)
#define LF_W      148
#define LF_H      ((int)(LUNG_BOT - LUNG_TOP) + 16)
#define HEART_SPR 120   /* sprite size; heart drawn at 1:1 around its centre */

static void build_heart(void)
{
    heart_spr = cairo_image_surface_create(CAIRO_FORMAT_A8, HEART_SPR, HEART_SPR);
    cairo_t *c = cairo_create(heart_spr);
    cairo_translate(c, HEART_SPR / 2.0, HEART_SPR / 2.0 + 6);

    /* great vessels: vena cava, aortic arch, pulmonary trunk, as hollow tubes */
    cairo_set_line_cap(c, CAIRO_LINE_CAP_ROUND);
    for (int pass = 0; pass < 2; pass++) {
        cairo_set_source_rgba(c, 0, 0, 0, pass ? 0.25 : 0.75);
        cairo_set_line_width(c, pass ? 4 : 8);
        cairo_move_to(c, -16, -16); cairo_curve_to(c, -18, -30, -17, -42, -16, -50); cairo_stroke(c);
        cairo_move_to(c, -2, -18); cairo_curve_to(c, -4, -36, 0, -46, 10, -46);
        cairo_curve_to(c, 20, -46, 24, -38, 22, -26); cairo_stroke(c);
        cairo_move_to(c, 8, -16); cairo_curve_to(c, 10, -26, 2, -32, -8, -34); cairo_stroke(c);
    }
    /* body: dense at the rim like on film, softer inside */
    heart_body(c);
    cairo_pattern_t *g = cairo_pattern_create_radial(2, 6, 4, 2, 6, 40);
    cairo_pattern_add_color_stop_rgba(g, 0, 0, 0, 0, 0.14);
    cairo_pattern_add_color_stop_rgba(g, 0.6, 0, 0, 0, 0.26);
    cairo_pattern_add_color_stop_rgba(g, 1, 0, 0, 0, 0.6);
    cairo_set_source(c, g);
    cairo_fill_preserve(c);
    cairo_pattern_destroy(g);
    cairo_set_source_rgba(c, 0, 0, 0, 1);
    cairo_set_line_width(c, 2);
    cairo_stroke(c);
    /* septum and coronary arteries */
    cairo_set_line_width(c, 1.4);
    cairo_set_source_rgba(c, 0, 0, 0, 0.9);
    cairo_move_to(c, 2, -18); cairo_curve_to(c, 0, 0, 8, 18, 24, 28); cairo_stroke(c);
    cairo_set_line_width(c, 1.0);
    cairo_set_source_rgba(c, 0, 0, 0, 0.6);
    cairo_move_to(c, -18, -10); cairo_curve_to(c, -24, 4, -14, 18, -2, 24); cairo_stroke(c);
    cairo_move_to(c, 14, -12); cairo_curve_to(c, 22, -2, 26, 10, 30, 18); cairo_stroke(c);
    cairo_move_to(c, 6, 4); cairo_curve_to(c, 14, 8, 16, 14, 20, 22); cairo_stroke(c);
    cairo_destroy(c);
}

/* Brain folds: wandering lines inside each hemisphere, drawn with (ox, oy) as the origin */
static void brain_folds(cairo_t *c, double ox, double oy, double alpha)
{
    srand(99);
    for (int side = 0; side < 2; side++) {
        double bx = CX + (side ? 14 : -14) - ox, by = BRAIN_Y - oy;
        cairo_save(c);
        cairo_new_path(c);
        cairo_save(c);
        cairo_translate(c, bx, by);
        cairo_scale(c, 13, 30);
        cairo_arc(c, 0, 0, 1, 0, 2 * M_PI);
        cairo_restore(c);
        cairo_set_source_rgba(c, 0, 0, 0, alpha);
        cairo_set_line_width(c, 1.2);
        cairo_stroke_preserve(c);
        cairo_clip(c);
        cairo_set_source_rgba(c, 0, 0, 0, alpha * 0.85);
        cairo_set_line_width(c, 1.1);
        for (int k = 0; k < 16; k++) {
            double x = bx + (frand() - 0.5) * 22, y = by + (frand() - 0.5) * 56, a = frand() * 2 * M_PI;
            cairo_move_to(c, x, y);
            for (int st = 0; st < 22; st++) {
                a += sin(st * 0.9 + k) * 0.9;
                x += cos(a) * 2.2, y += sin(a) * 2.2;
                cairo_line_to(c, x, y);
            }
            cairo_stroke(c);
        }
        cairo_restore(c);
    }
}

static void build_static(void)
{
    srand(11);

    /* ---- background */
    bg_surf = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cairo_t *c = cairo_create(bg_surf);
    cairo_pattern_t *g = cairo_pattern_create_radial(CX, CY - 20, 10, CX, CY, 250);
    cairo_pattern_add_color_stop_rgb(g, 0, 0.030, 0.050, 0.075);
    cairo_pattern_add_color_stop_rgb(g, 0.7, 0.012, 0.020, 0.034);
    cairo_pattern_add_color_stop_rgb(g, 1, 0.004, 0.006, 0.010);
    cairo_set_source(c, g);
    cairo_paint(c);
    cairo_pattern_destroy(g);
    cairo_destroy(c);
    /* film grain */
    cairo_surface_flush(bg_surf);
    unsigned char *px = cairo_image_surface_get_data(bg_surf);
    int bst = cairo_image_surface_get_stride(bg_surf);
    for (int y = 0; y < SIZE; y++)
        for (int x = 0; x < SIZE; x++) {
            int n = (int)(frand() * 7) - 3;
            for (int k = 0; k < 3; k++) {
                int v = px[y * bst + x * 4 + k] + n;
                px[y * bst + x * 4 + k] = (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v);
            }
        }
    cairo_surface_mark_dirty(bg_surf);

    /* ---- bones: luminance of the image becomes alpha, plus a soft glow and bright cores */
    cairo_surface_t *img = load_asset("body.png");
    cairo_surface_t *src = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    c = cairo_create(src);
    cairo_scale(c, (double)SIZE / cairo_image_surface_get_width(img), (double)SIZE / cairo_image_surface_get_height(img));
    cairo_set_source_surface(c, img, 0, 0);
    cairo_paint(c);
    cairo_destroy(c);
    cairo_surface_destroy(img);
    cairo_surface_flush(src);
    unsigned char *sp = cairo_image_surface_get_data(src);
    int sst = cairo_image_surface_get_stride(src);

    fig_mask = cairo_image_surface_create(CAIRO_FORMAT_A8, SIZE, SIZE);
    core_mask = cairo_image_surface_create(CAIRO_FORMAT_A8, SIZE, SIZE);
    cairo_surface_flush(fig_mask);
    cairo_surface_flush(core_mask);
    unsigned char *fm = cairo_image_surface_get_data(fig_mask), *cm = cairo_image_surface_get_data(core_mask);
    int ast = cairo_image_surface_get_stride(fig_mask);
    unsigned char *glow = calloc(ast * SIZE, 1);
    for (int y = 0; y < SIZE; y++) {
        double fade = 1 - 0.7 * smoothstep(372, 440, y);         /* let the ECG read over the pelvis */
        for (int x = 0; x < SIZE; x++) {
            const unsigned char *p = sp + y * sst + x * 4;
            double l = (p[0] * 0.2 + p[1] * 0.5 + p[2] * 0.3) / 255.0;
            l = l < 0.06 ? 0 : (l - 0.06) / 0.94;               /* crush the black */
            glow[y * ast + x] = (unsigned char)(255 * l * fade);
            fm[y * ast + x] = (unsigned char)(255 * l * fade * 0.62);
            cm[y * ast + x] = (unsigned char)(255 * pow(smoothstep(0.5, 1.0, l), 1.3) * fade * 0.8);
        }
    }
    blur_a8(glow, SIZE, SIZE, ast, 5);
    for (int i = 0; i < ast * SIZE; i++) {
        int v = fm[i] + glow[i] * 5 / 10;
        fm[i] = (unsigned char)(v > 255 ? 255 : v);
    }
    free(glow);
    cairo_surface_destroy(src);
    cairo_surface_mark_dirty(fig_mask);
    cairo_surface_mark_dirty(core_mask);

    /* ---- the rest of the figure, drawn straight into the same mask */
    c = cairo_create(fig_mask);
    cairo_set_line_cap(c, CAIRO_LINE_CAP_ROUND);
    cairo_set_line_join(c, CAIRO_LINE_JOIN_ROUND);

    /* haze in the body cavity */
    g = cairo_pattern_create_radial(CX, 250, 10, CX, 250, 150);
    cairo_pattern_add_color_stop_rgba(g, 0, 0, 0, 0, 0.07);
    cairo_pattern_add_color_stop_rgba(g, 1, 0, 0, 0, 0);
    cairo_set_source(c, g);
    cairo_arc(c, CX, 250, 150, 0, 2 * M_PI);
    cairo_fill(c);
    cairo_pattern_destroy(g);

    /* lungs: soft edge and a faint bronchial tree */
    for (int side = 0; side < 2; side++) {
        lung_shape(c, side);
        lung_path[side] = cairo_copy_path(c);
        cairo_set_source_rgba(c, 0, 0, 0, 0.35);
        cairo_set_line_width(c, 1.4);
        cairo_stroke(c);
    }
    cairo_set_source_rgba(c, 0, 0, 0, 0.30);
    cairo_set_line_width(c, 5);
    cairo_move_to(c, CX, 138); cairo_line_to(c, CX, 204); cairo_stroke(c);   /* trachea */
    cairo_save(c);
    cairo_append_path(c, lung_path[0]);
    cairo_append_path(c, lung_path[1]);
    cairo_clip(c);
    cairo_set_source_rgba(c, 0, 0, 0, 0.16);
    srand(77);
    bronchus(c, CX - 2, 204, M_PI * 0.72, 26, 3.5, 7);
    bronchus(c, CX + 2, 204, M_PI * 0.28, 22, 3.5, 7);
    cairo_restore(c);

    /* the stomach wall and the gut */
    stomach_shape(c);
    stom_path = cairo_copy_path(c);
    cairo_set_source_rgba(c, 0, 0, 0, 0.45);
    cairo_set_line_width(c, 1.6);
    cairo_stroke(c);
    static const double gut_pts[][2] = {
        { STOM_X - 34, STOM_Y + 10 }, { 222, 352 }, { 214, 366 }, { 232, 374 }, { 262, 366 }, { 280, 378 },
        { 266, 392 }, { 236, 390 }, { 214, 396 }, { 222, 410 }, { 252, 408 },
    };
    make_vpath(&gut, gut_pts, (int)(sizeof(gut_pts) / sizeof(gut_pts[0])));
    cairo_set_source_rgba(c, 0, 0, 0, 0.30);
    cairo_set_line_width(c, 1.1);
    vpath_stroke(c, &gut, -3.5);
    vpath_stroke(c, &gut, 3.5);

    /* vessels: a vein and an artery side by side */
    static const double larm[][2] = { { HEART_X - 20, 236 }, { 226, 206 }, { 190, 190 }, { 148, 200 }, { 132, 250 },
                                      { 122, 318 }, { 112, 360 }, { 102, 398 } };
    static const double rarm[][2] = { { HEART_X + 8, 226 }, { 262, 204 }, { 296, 190 }, { 334, 200 }, { 348, 250 },
                                      { 357, 318 }, { 366, 360 }, { 377, 398 } };
    static const double neck[][2] = { { HEART_X - 4, 220 }, { 254, 190 }, { 256, 160 }, { 253, 132 }, { 248, 112 } };
    static const double aorta[][2] = { { HEART_X + 2, 282 }, { 248, 310 }, { 244, 350 }, { 240, 384 }, { 240, 402 } };
    make_vpath(&vessels[V_LARM], larm, 8);
    make_vpath(&vessels[V_RARM], rarm, 8);
    make_vpath(&vessels[V_NECK], neck, 5);
    make_vpath(&vessels[V_AORTA], aorta, 5);
    cairo_set_line_width(c, 1.0);
    cairo_set_source_rgba(c, 0, 0, 0, 0.28);
    for (int i = 0; i < N_VESSELS; i++) {
        vpath_stroke(c, &vessels[i], -3);
        vpath_stroke(c, &vessels[i], 3);
    }

    brain_folds(c, 0, 0, 0.35);

    /* spinal cord and the intercostal nerves that fan out from it along the ribs */
    static const double cord[][2] = { { CX, 118 }, { CX, 250 }, { CX, 398 } };
    make_vpath(&spine, cord, 3);
    cairo_set_source_rgba(c, 0, 0, 0, 0.12);
    cairo_set_line_width(c, 0.9);
    for (int i = 0; i < N_NERVES; i++) {
        double m = i % 2 ? 1 : -1, y = 196 + (i / 2) * 25;
        double pts[][2] = { { CX + m * 5, y }, { CX + m * 30, y - 5 }, { CX + m * 56, y + 4 }, { CX + m * 70, y + 20 },
                            { CX + m * 74, y + 34 } };
        make_vpath(&nerves[i], (const double (*)[2])pts, 5);
        vpath_stroke(c, &nerves[i], 0);
    }
    /* bezel: fine ticks round the rim, gauge tracks, ECG baseline */
    cairo_set_source_rgba(c, 0, 0, 0, 0.35);
    cairo_set_line_width(c, 1);
    cairo_arc(c, CX, CY, 237, 0, 2 * M_PI);
    cairo_stroke(c);
    for (int i = 0; i < 120; i++) {
        double a = i * 2 * M_PI / 120, r0 = i % 10 ? 232 : 228;
        if (i > 76 && i < 104 && (i < 88 || i > 92))
            continue;                   /* room for the rim text */
        cairo_set_source_rgba(c, 0, 0, 0, i % 10 ? 0.22 : 0.45);
        cairo_move_to(c, CX + cos(a) * r0, CY + sin(a) * r0);
        cairo_line_to(c, CX + cos(a) * 236, CY + sin(a) * 236);
        cairo_stroke(c);
    }
    cairo_set_source_rgba(c, 0, 0, 0, 0.18);
    cairo_set_line_width(c, 4);
    cairo_arc(c, CX, CY, GAUGE_R - 8, M_PI * 0.72, M_PI * 1.28);
    cairo_stroke(c);
    cairo_new_path(c);
    cairo_arc(c, CX, CY, GAUGE_R - 8, -M_PI * 0.28, M_PI * 0.28);
    cairo_stroke(c);
    cairo_set_source_rgba(c, 0, 0, 0, 0.14);
    cairo_set_line_width(c, 1);
    for (int i = 0; i <= 8; i++) {
        double x = ECG_X0 + (ECG_X1 - ECG_X0) * i / 8;
        cairo_move_to(c, x, ECG_Y - ECG_AMP - 6);
        cairo_line_to(c, x, ECG_Y + ECG_AMP * 0.5 + 4);
    }
    cairo_stroke(c);
    cairo_destroy(c);

    /* the full lungs: a dense, textured fill revealed from below as RAM fills */
    brain_spr = cairo_image_surface_create(CAIRO_FORMAT_A8, 80, 90);
    c = cairo_create(brain_spr);
    brain_folds(c, BRAIN_X0, BRAIN_Y0, 1.0);
    cairo_destroy(c);

    lung_fill_mask = cairo_image_surface_create(CAIRO_FORMAT_A8, LF_W, LF_H);
    c = cairo_create(lung_fill_mask);
    cairo_translate(c, -LF_X0, -LF_Y0);
    for (int side = 0; side < 2; side++) {
        double lx = CX + (side ? 36 : -36);
        cairo_new_path(c);
        cairo_append_path(c, lung_path[side]);
        g = cairo_pattern_create_radial(lx, 250, 10, lx, 255, 80);
        cairo_pattern_add_color_stop_rgba(g, 0, 0, 0, 0, 0.18);
        cairo_pattern_add_color_stop_rgba(g, 1, 0, 0, 0, 0.42);
        cairo_set_source(c, g);
        cairo_fill_preserve(c);
        cairo_pattern_destroy(g);
        cairo_set_source_rgba(c, 0, 0, 0, 0.8);
        cairo_set_line_width(c, 1.8);
        cairo_stroke(c);
    }
    cairo_save(c);
    cairo_append_path(c, lung_path[0]);
    cairo_append_path(c, lung_path[1]);
    cairo_clip(c);
    /* alveoli: little rings */
    cairo_set_line_width(c, 0.8);
    for (int k = 0; k < 520; k++) {
        double x = CX - 70 + frand() * 140, y = LUNG_TOP + frand() * (LUNG_BOT - LUNG_TOP);
        cairo_new_sub_path(c);
        cairo_arc(c, x, y, 1.5 + frand() * 2.5, 0, 2 * M_PI);
    }
    cairo_set_source_rgba(c, 0, 0, 0, 0.22);
    cairo_stroke(c);
    srand(77);          /* same tree as the faint one */
    cairo_set_source_rgba(c, 0, 0, 0, 0.55);
    cairo_set_line_cap(c, CAIRO_LINE_CAP_ROUND);
    bronchus(c, CX - 2, 204, M_PI * 0.72, 26, 3.5, 7);
    bronchus(c, CX + 2, 204, M_PI * 0.28, 22, 3.5, 7);
    cairo_restore(c);
    cairo_destroy(c);

    build_heart();
    for (int i = 0; i < N_TWITCH; i++)
        twitches[i].age = 1;

    rim_mask = cairo_image_surface_create(CAIRO_FORMAT_A8, SIZE, SIZE);
    c = cairo_create(rim_mask);
    g = cairo_pattern_create_radial(CX, CY, 170, CX, CY, 240);
    cairo_pattern_add_color_stop_rgba(g, 0, 0, 0, 0, 0);
    cairo_pattern_add_color_stop_rgba(g, 1, 0, 0, 0, 0.22);
    cairo_set_source(c, g);
    cairo_arc(c, CX, CY, 240, 0, 2 * M_PI);
    cairo_fill(c);
    cairo_pattern_destroy(g);
    cairo_destroy(c);

    hud_mask = cairo_image_surface_create(CAIRO_FORMAT_A8, SIZE, SIZE);
    hud_shadow = cairo_image_surface_create(CAIRO_FORMAT_A8, SIZE, SIZE);
}

/* ---- per-frame state */

/* Heart envelope: lub at 0, dub at 0.16 of the beat */
static double pulse_env(double ph)
{
    ph -= floor(ph);
    double a = exp(-pow(ph / 0.06, 2)) + exp(-pow((ph - 1) / 0.06, 2));
    double b = exp(-pow((ph - 0.17) / 0.05, 2));
    return fmin(1, a + 0.55 * b);
}

static double bpm_of(double cpu) { return 52 + 118 * cpu; }

/* rates to a 0..1 activity on a log scale, so a trickle still shows */
static double act(double bps, double full)
{
    return clamp01(log1p(bps / 2e4) / log1p(full / 2e4));
}

static double gpu_act(const stats *g, int i)
{
    double pw = clamp01((g->power[i] - GPU_IDLE_W) / (GPU_MAX_W - GPU_IDLE_W));
    double a = 0.5 * clamp01(g->load[i]) + 0.5 * pw;
    return a < 0.03 ? 0 : a;
}

static void simulate(const stats *g, shown_t *sh, double dt, double t)
{
    double prev = beat_phase;
    beat_phase += dt * bpm_of(sh->cpu) / 60.0;
    if (floor(beat_phase) != floor(prev)) {          /* a new beat: when did it land? */
        double f = (floor(beat_phase) - prev) / (beat_phase - prev);
        memmove(beat_t + 1, beat_t, sizeof(double) * (N_BEATS - 1));
        beat_t[0] = t - dt * (1 - f);
        if (beat_n < N_BEATS)
            beat_n++;
    }
    breath_phase += dt * (0.2 + 0.1 * sh->cpu + 0.55 * sh->psi);     /* pressure: laboured breathing */

    /* page faults twitch the nerves; major faults twitch hard */
    twitch_acc += dt * (1.5 * sh->pf + 22 * pow(sh->pf, 2.2) + 6 * sh->pfmaj);
    while (twitch_acc >= 1) {
        twitch_acc -= 1;
        for (int i = 0; i < N_TWITCH; i++)
            if (twitches[i].age >= 0.6) {
                twitches[i] = (twitch){ rand() % N_NERVES, 0, frand() < sh->pfmaj ? 1 : 0.3 + 0.4 * frand() };
                break;
            }
    }
    for (int i = 0; i < N_TWITCH; i++)
        twitches[i].age += dt;

    /* interrupts: small blips on the ECG between heartbeats */
    blip_acc += dt * (1 + 24 * pow(sh->intr, 1.5));
    while (blip_acc >= 1) {
        blip_acc -= 1;
        memmove(blip_t + 1, blip_t, sizeof(double) * (N_BLIPS - 1));
        memmove(blip_a + 1, blip_a, sizeof(double) * (N_BLIPS - 1));
        blip_t[0] = t - frand() * dt;
        blip_a[0] = (frand() < 0.5 ? -1 : 1) * (0.04 + (0.06 + 0.14 * frand()) * sh->intr);
        if (blip_n < N_BLIPS)
            blip_n++;
    }
    churn += dt * (0.4 + 7.0 * sh->rd);
    flow_rx += dt * (18 + 150 * sh->rx);
    flow_tx += dt * (18 + 150 * sh->tx);
    flow_gut += dt * (6 + 70 * sh->wr);

    /* brain sparks: GPU activity (and vLLM tokens, if any) */
    for (int side = 0; side < 2; side++) {
        double tok = g->tok_port[side] > 1 ? clamp01(g->tok_port[side] / 600) : 0;
        spark_acc[side] += dt * (26 * sh->gpu[side] + 10 * tok);
        while (spark_acc[side] >= 1) {
            spark_acc[side] -= 1;
            for (int i = 0; i < N_SPARKS; i++)
                if (sparks[i].age >= sparks[i].life) {
                    double a = frand() * 2 * M_PI, r = sqrt(frand()) * 0.85;
                    sparks[i] = (spark){ CX + (side ? 14 : -14) + cos(a) * r * 12, BRAIN_Y + sin(a) * r * 28,
                                         0, 0.25 + frand() * 0.35, side };
                    break;
                }
        }
    }
    for (int i = 0; i < N_SPARKS; i++)
        sparks[i].age += dt;
}

/* ---- drawing */

static void glow_disc(cairo_t *cr, double x, double y, double r, rgb c, double a)
{
    if (a <= 0.004)
        return;
    cairo_pattern_t *g = cairo_pattern_create_radial(x, y, 0, x, y, r);
    cairo_pattern_add_color_stop_rgba(g, 0, c.r, c.g, c.b, a);
    cairo_pattern_add_color_stop_rgba(g, 0.4, c.r, c.g, c.b, a * 0.4);
    cairo_pattern_add_color_stop_rgba(g, 1, c.r, c.g, c.b, 0);
    cairo_set_source(cr, g);
    cairo_rectangle(cr, x - r, y - r, 2 * r, 2 * r);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
}

static void draw_lungs(cairo_t *cr, const shown_t *sh, rgb tint, double t)
{
    double level = LUNG_BOT - (LUNG_BOT - LUNG_TOP + 4) * clamp01(sh->ram);
    double breath = 0.5 - 0.5 * cos(breath_phase * 2 * M_PI);
    double sc = 1 + (0.02 + 0.045 * sh->psi) * breath;

    cairo_save(cr);
    cairo_translate(cr, CX, LUNG_TOP);         /* breathe from the top */
    cairo_scale(cr, sc, sc);
    cairo_translate(cr, -CX, -LUNG_TOP);
    /* filled part */
    cairo_save(cr);
    cairo_new_path(cr);
    cairo_move_to(cr, CX - 80, SIZE);
    for (int x = -80; x <= 80; x += 8)
        cairo_line_to(cr, CX + x, level + 2.2 * sin(x * 0.09 + t * 2.2));
    cairo_line_to(cr, CX + 80, SIZE);
    cairo_close_path(cr);
    cairo_clip(cr);
    set_rgba(cr, tint, 0.55 + 0.25 * breath);
    cairo_mask_surface(cr, lung_fill_mask, LF_X0, LF_Y0);
    cairo_restore(cr);
    /* the bright surface line */
    if (sh->ram > 0.01) {
        cairo_save(cr);
        cairo_new_path(cr);
        cairo_append_path(cr, lung_path[0]);
        cairo_append_path(cr, lung_path[1]);
        cairo_clip(cr);
        cairo_new_path(cr);
        for (int x = -80; x <= 80; x += 4)
            cairo_line_to(cr, CX + x, level + 2.2 * sin(x * 0.09 + t * 2.2));
        set_rgba(cr, pale(tint, 0.3), 0.35);
        cairo_set_line_width(cr, 5);
        cairo_stroke_preserve(cr);
        set_rgba(cr, pale(tint, 0.6), 0.8);
        cairo_set_line_width(cr, 1.4);
        cairo_stroke(cr);
        cairo_restore(cr);
    }
    cairo_restore(cr);
}

static void draw_heart(cairo_t *cr, const shown_t *sh, rgb hc)
{
    double p = pulse_env(beat_phase), load = sh->cpu;
    double bright = 0.5 + 0.2 * load + 0.3 * p;

    glow_disc(cr, HEART_X, HEART_Y, 56 + 26 * load + 12 * p, hc, 0.07 + 0.12 * load + 0.16 * p);

    /* thread ring: one tick per hardware thread, length and brightness = its load */
    int n = 32;
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    cairo_set_line_width(cr, 2.6);
    for (int pass = 0; pass < 2; pass++) {
        for (int i = 0; i < n; i++) {
            double l = sh->thread[i], a = -M_PI / 2 + i * 2 * M_PI / n;
            double r0 = 42 + 2.5 * p, r1 = r0 + 3 + 13 * l;
            if ((pass == 0) != (l < 0.5))
                continue;
            cairo_move_to(cr, HEART_X + cos(a) * r0, HEART_Y + sin(a) * r0);
            cairo_line_to(cr, HEART_X + cos(a) * r1, HEART_Y + sin(a) * r1);
        }
        set_rgba(cr, pass ? pale(hc, 0.35) : hc, pass ? 0.95 : 0.45);
        cairo_stroke(cr);
    }

    double sc = 1 + 0.09 * p + 0.03 * load;
    cairo_save(cr);
    cairo_translate(cr, HEART_X, HEART_Y);
    cairo_scale(cr, sc, sc);
    set_rgba(cr, hc, fmin(1, bright));
    cairo_mask_surface(cr, heart_spr, -HEART_SPR / 2.0, -HEART_SPR / 2.0 - 6);
    set_rgba(cr, pale(hc, 0.6), 0.28 * p);
    cairo_mask_surface(cr, heart_spr, -HEART_SPR / 2.0, -HEART_SPR / 2.0 - 6);
    cairo_restore(cr);
}

static void draw_brain(cairo_t *cr, const shown_t *sh, rgb tint, double t)
{
    for (int side = 0; side < 2; side++) {
        double a = sh->gpu[side], bx = CX + (side ? 14 : -14);
        double shimmer = 0.85 + 0.15 * sin(t * 5 + side * 2);
        glow_disc(cr, bx, BRAIN_Y, 34 + 14 * a, tint, 0.05 + 0.32 * a * shimmer);
        cairo_save(cr);
        cairo_rectangle(cr, side ? CX : BRAIN_X0, BRAIN_Y0, 40, 90);
        cairo_clip(cr);
        set_rgba(cr, pale(tint, 0.2), 0.10 + 0.75 * a * shimmer);
        cairo_mask_surface(cr, brain_spr, BRAIN_X0, BRAIN_Y0);
        cairo_restore(cr);
    }
    /* sparks: tiny flashes, each linked to the one before it like neurons firing */
    rgb sc = pale(tint, 0.55);
    int prev[2] = { -1, -1 };
    cairo_new_path(cr);
    for (int i = 0; i < N_SPARKS; i++) {
        spark *s = &sparks[i];
        if (s->age >= s->life)
            continue;
        if (prev[s->side] >= 0) {
            cairo_move_to(cr, sparks[prev[s->side]].x, sparks[prev[s->side]].y);
            cairo_line_to(cr, s->x, s->y);
        }
        prev[s->side] = i;
    }
    set_rgba(cr, sc, 0.22);
    cairo_set_line_width(cr, 0.8);
    cairo_stroke(cr);
    for (int pass = 0; pass < 2; pass++) {
        cairo_new_path(cr);
        for (int i = 0; i < N_SPARKS; i++) {
            spark *s = &sparks[i];
            if (s->age >= s->life)
                continue;
            double k = sin(s->age / s->life * M_PI);
            cairo_new_sub_path(cr);
            cairo_arc(cr, s->x, s->y, pass ? 0.6 + 1.0 * k : 2 + 3 * k, 0, 2 * M_PI);
        }
        set_rgba(cr, sc, pass ? 0.9 : 0.16);
        cairo_fill(cr);
    }
}

static void draw_gut(cairo_t *cr, const shown_t *sh, rgb tint)
{
    double a = fmax(sh->rd, sh->wr);
    glow_disc(cr, STOM_X - 4, STOM_Y + 2, 44 + 16 * a, tint, 0.03 + 0.2 * sh->rd);

    /* stomach churning with reads: a swirl of particles and a bright wall */
    cairo_save(cr);
    cairo_new_path(cr);
    cairo_append_path(cr, stom_path);
    set_rgba(cr, tint, 0.06 + 0.22 * sh->rd);
    cairo_fill_preserve(cr);
    set_rgba(cr, pale(tint, 0.3), 0.2 + 0.6 * sh->rd);
    cairo_set_line_width(cr, 1.5);
    cairo_stroke(cr);
    int n = 10 + (int)(22 * sh->rd);
    for (int i = 0; i < n; i++) {
        double r = 6 + (i * 37 % 19), ang = churn * (1.2 - r / 40) + i * 2.399;
        double x = STOM_X + 2 + cos(ang) * r * 1.1, y = STOM_Y - 4 + sin(ang) * r * 0.75;
        cairo_new_sub_path(cr);
        cairo_arc(cr, x, y, 1.1 + 0.9 * sh->rd, 0, 2 * M_PI);
    }
    set_rgba(cr, pale(tint, 0.4), 0.25 + 0.65 * sh->rd);
    cairo_fill(cr);
    cairo_restore(cr);

    /* gut: writes flow down it */
    double len = (gut.n - 1) * VSTEP, tail = 5 + 9 * sh->wr;
    int m = (int)(3 + 14 * sh->wr);
    cairo_new_path(cr);
    for (int i = 0; i < m; i++) {
        double d = fmod(flow_gut + i * len / m, len), x, y;
        vpath_at(&gut, d, 0, &x, &y);
        cairo_move_to(cr, x, y);
        vpath_at(&gut, d - tail < 0 ? 0 : d - tail, 0, &x, &y);
        cairo_line_to(cr, x, y);
    }
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    set_rgba(cr, pale(tint, 0.35), 0.1 + 0.15 * sh->wr);
    cairo_set_line_width(cr, 5);
    cairo_stroke_preserve(cr);
    set_rgba(cr, pale(tint, 0.35), 0.3 + 0.65 * sh->wr);
    cairo_set_line_width(cr, 1.8);
    cairo_stroke(cr);
}

/* Nerves: a page-fault twitch runs out from the spine along a rib */
static void draw_nerves(cairo_t *cr, rgb tint)
{
    rgb c = pale(tint, 0.6);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    for (int i = 0; i < N_TWITCH; i++) {
        twitch *w = &twitches[i];
        if (w->age >= 0.6)
            continue;
        const vpath *v = &nerves[w->nerve];
        double len = (v->n - 1) * VSTEP, head = w->age * 420, fade = 1 - w->age / 0.6, x, y;
        if (w->age < 0.15) {             /* the whole nerve flickers as it fires */
            set_rgba(cr, c, (0.25 + 0.35 * w->strong) * (1 - w->age / 0.15));
            cairo_set_line_width(cr, 1 + w->strong);
            vpath_stroke(cr, v, 0);
        }
        if (head - 18 > len)
            continue;
        cairo_new_path(cr);
        vpath_at(v, fmin(head, len), 0, &x, &y);
        cairo_move_to(cr, x, y);
        vpath_at(v, fmax(0, head - 18), 0, &x, &y);
        cairo_line_to(cr, x, y);
        set_rgba(cr, c, 0.9 * fade);
        cairo_set_line_width(cr, 1.4 + 1.4 * w->strong);
        cairo_stroke(cr);
    }
}

/* Spinal cord: PCIe traffic, host to GPU climbing to the brain, GPU to host running down */
static void draw_spine(cairo_t *cr, const shown_t *sh, rgb tint, double t)
{
    double len = (spine.n - 1) * VSTEP, a = fmax(sh->pcie_up, sh->pcie_dn);
    rgb c = pale(tint, 0.5);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    if (a > 0.01) {
        cairo_move_to(cr, CX, spine.y[0]);
        cairo_line_to(cr, CX, spine.y[spine.n - 1]);
        set_rgba(cr, tint, 0.3 * a);
        cairo_set_line_width(cr, 4);
        cairo_stroke(cr);
    }
    for (int dir = 0; dir < 2; dir++) {
        double k = dir ? sh->pcie_dn : sh->pcie_up;
        int m = (int)(10 * k + 0.5);
        if (m == 0)
            continue;
        double speed = 120 + 380 * k, tail = 10 + 26 * k;
        cairo_new_path(cr);
        for (int i = 0; i < m; i++) {
            double d = fmod(t * speed + i * len / m + dir * 17, len);
            double y0 = dir ? spine.y[0] + d : spine.y[spine.n - 1] - d;
            double y1 = dir ? fmax(spine.y[0], y0 - tail) : fmin(spine.y[spine.n - 1], y0 + tail);
            cairo_move_to(cr, CX + (dir ? 1.2 : -1.2), y0);
            cairo_line_to(cr, CX + (dir ? 1.2 : -1.2), y1);
        }
        set_rgba(cr, c, 0.25 + 0.7 * k);
        cairo_set_line_width(cr, 2);
        cairo_stroke(cr);
    }
}

/* Blood: download flows in along the veins, upload is pumped out along the arteries */
static void draw_blood(cairo_t *cr, const shown_t *sh, rgb tint, rgb hc)
{
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    for (int dir = 0; dir < 2; dir++) {
        double a = dir ? sh->tx : sh->rx, flow = dir ? flow_tx : flow_rx, tail = 4 + 10 * a;
        rgb c = dir ? pale(hc, 0.2) : pale(tint, 0.35);
        cairo_new_path(cr);
        for (int v = 0; v < N_VESSELS; v++) {
            double len = (vessels[v].n - 1) * VSTEP;
            int m = 1 + (int)(len / 50 * (0.2 + 1.8 * a));
            for (int i = 0; i < m; i++) {
                double d = fmod(flow + i * len / m + v * 13 + (i * 7919 % 13), len), x, y;
                double d2 = d - tail < 0 ? 0 : d - tail;
                if (!dir)
                    d = len - d, d2 = len - d2;          /* veins run to the heart */
                vpath_at(&vessels[v], d, dir ? -3 : 3, &x, &y);
                cairo_move_to(cr, x, y);
                vpath_at(&vessels[v], d2, dir ? -3 : 3, &x, &y);
                cairo_line_to(cr, x, y);
            }
        }
        cairo_set_line_cap(cr, CAIRO_LINE_CAP_BUTT);
        set_rgba(cr, c, 0.08 + 0.14 * a);
        cairo_set_line_width(cr, 4);
        cairo_stroke_preserve(cr);
        cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
        set_rgba(cr, c, 0.35 + 0.6 * a);
        cairo_set_line_width(cr, 1.5);
        cairo_stroke(cr);
    }
}

/* ECG: P, QRS and T waves around each recorded beat, scrolling left */
static double ecg_at(double tt)
{
    double v = 0;
    for (int i = 0; i < blip_n; i++) {
        double d = tt - blip_t[i];
        if (d < -0.02)
            continue;
        if (d > 0.02)
            break;
        v += blip_a[i] * exp(-pow(d / 0.005, 2) / 2);
    }
    for (int i = 0; i < beat_n; i++) {
        double d = tt - beat_t[i];
        if (d < -0.3)
            continue;
        if (d > 0.6)
            break;
        double rr = i + 1 < beat_n ? beat_t[i] - beat_t[i + 1] : 1.0, s = fmin(1, rr / 0.8);
        v += 0.12 * exp(-pow((d + 0.16 * s) / 0.025, 2) / 2)
           - 0.14 * exp(-pow((d + 0.028) / 0.009, 2) / 2)
           + 1.00 * exp(-pow(d / 0.016, 2) / 2)
           - 0.32 * exp(-pow((d - 0.03) / 0.011, 2) / 2)
           + 0.24 * exp(-pow((d - 0.27 * s) / 0.045, 2) / 2);
    }
    return v;
}

static void draw_ecg(cairo_t *cr, rgb hc, double t)
{
    cairo_new_path(cr);
    double hx = ECG_X1, hy = ECG_Y;
    for (double x = ECG_X0; x <= ECG_X1; x += 1) {
        double y = ECG_Y - ECG_AMP * ecg_at(t - (ECG_X1 - x) / ECG_SPEED);
        if (x == ECG_X0)
            cairo_move_to(cr, x, y);
        else
            cairo_line_to(cr, x, y);
        hy = y;
    }
    cairo_pattern_t *g = cairo_pattern_create_linear(ECG_X0, 0, ECG_X1, 0);
    cairo_pattern_add_color_stop_rgba(g, 0, hc.r, hc.g, hc.b, 0);
    cairo_pattern_add_color_stop_rgba(g, 0.5, hc.r, hc.g, hc.b, 0.12);
    cairo_pattern_add_color_stop_rgba(g, 1, hc.r, hc.g, hc.b, 0.3);
    cairo_set_source(cr, g);
    cairo_set_line_width(cr, 5);
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_BEVEL);
    cairo_stroke_preserve(cr);
    cairo_pattern_destroy(g);
    rgb p = pale(hc, 0.45);
    g = cairo_pattern_create_linear(ECG_X0, 0, ECG_X1, 0);
    cairo_pattern_add_color_stop_rgba(g, 0, p.r, p.g, p.b, 0);
    cairo_pattern_add_color_stop_rgba(g, 0.35, p.r, p.g, p.b, 0.6);
    cairo_pattern_add_color_stop_rgba(g, 1, p.r, p.g, p.b, 1);
    cairo_set_source(cr, g);
    cairo_set_line_width(cr, 1.8);
    cairo_stroke(cr);
    cairo_pattern_destroy(g);
    glow_disc(cr, hx, hy, 12, pale(hc, 0.5), 0.9);
}

/* Rim gauges: CPU on the left, the fever thermometer on the right, both filling upward */
static void draw_gauges(cairo_t *cr, const shown_t *sh, rgb tint, rgb hc)
{
    cairo_set_line_width(cr, 4);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    double span = M_PI * 0.56;
    double cpu = clamp01(sh->cpu), temp = clamp01((sh->tctl - 30) / 70);
    if (cpu > 0.005) {
        cairo_new_path(cr);
        /* left arc runs from the bottom (0.72 pi) up to the top (1.28 pi) */
        cairo_arc(cr, CX, CY, GAUGE_R - 8, M_PI * 0.72, M_PI * 0.72 + span * cpu);
        set_rgba(cr, hc, 0.9);
        cairo_stroke(cr);
    }
    if (temp > 0.005) {
        /* right arc from the bottom (0.28 pi) up to the top (-0.28 pi) */
        cairo_new_path(cr);
        cairo_arc_negative(cr, CX, CY, GAUGE_R - 8, M_PI * 0.28, M_PI * 0.28 - span * temp);
        set_rgba(cr, pale(tint, 0.15), 0.9);
        cairo_stroke(cr);
        double a = M_PI * 0.28 - span * temp;
        glow_disc(cr, CX + cos(a) * (GAUGE_R - 8), CY + sin(a) * (GAUGE_R - 8), 14, pale(tint, 0.3), 0.8);
    }
}

static void draw_scanline(cairo_t *cr, rgb tint, double t)
{
    double period = 5.5, u = fmod(t, period) / period;
    double y = -30 + u * (SIZE + 60), trail = 70;

    /* bones light up behind the line */
    cairo_pattern_t *g = cairo_pattern_create_linear(0, y - trail, 0, y);
    rgb p = pale(tint, 0.4);
    cairo_pattern_add_color_stop_rgba(g, 0, p.r, p.g, p.b, 0);
    cairo_pattern_add_color_stop_rgba(g, 1, p.r, p.g, p.b, 0.9);
    cairo_save(cr);
    cairo_rectangle(cr, 0, y - trail, SIZE, trail);
    cairo_clip(cr);
    cairo_set_source(cr, g);
    cairo_mask_surface(cr, fig_mask, 0, 0);
    cairo_restore(cr);
    cairo_pattern_destroy(g);

    /* the line itself, cut to the circle */
    double half = SIZE / 2.0 - 6, dy = y - CY;
    if (fabs(dy) >= half)
        return;
    double w = sqrt(half * half - dy * dy);
    g = cairo_pattern_create_linear(CX - w, 0, CX + w, 0);
    cairo_pattern_add_color_stop_rgba(g, 0, tint.r, tint.g, tint.b, 0);
    cairo_pattern_add_color_stop_rgba(g, 0.5, tint.r, tint.g, tint.b, 0.5);
    cairo_pattern_add_color_stop_rgba(g, 1, tint.r, tint.g, tint.b, 0);
    cairo_set_source(cr, g);
    cairo_rectangle(cr, CX - w, y - 1, 2 * w, 2);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    g = cairo_pattern_create_linear(0, y - 14, 0, y + 3);
    cairo_pattern_add_color_stop_rgba(g, 0, tint.r, tint.g, tint.b, 0);
    cairo_pattern_add_color_stop_rgba(g, 1, tint.r, tint.g, tint.b, 0.12);
    cairo_set_source(cr, g);
    cairo_rectangle(cr, CX - w, y - 14, 2 * w, 17);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
}

/* ---- text */

#define FONT "DejaVu Sans"

/* Largest size up to want that fits width w */
static double fit_size(cairo_t *c, const char *s, double want, double w)
{
    cairo_text_extents_t e;
    cairo_set_font_size(c, want);
    cairo_text_extents(c, s, &e);
    return e.x_advance > w ? want * w / e.x_advance : want;
}

/* align: 0 left, 1 right, 0.5 centre; the box never leaves the circle at that height */
static void hud_text(cairo_t *c, const char *s, double x, double y, double size, int bold, double align, double a)
{
    cairo_text_extents_t e;
    cairo_select_font_face(c, FONT, CAIRO_FONT_SLANT_NORMAL, bold ? CAIRO_FONT_WEIGHT_BOLD : CAIRO_FONT_WEIGHT_NORMAL);
    double top = y - size * 0.75, half = sqrt(fmax(0, 232.0 * 232 - pow(fmax(fabs(top - CY), fabs(y - CY)), 2)));
    double room = align == 1 ? x - (CX - half) : align == 0 ? CX + half - x : 2 * fmin(x - (CX - half), CX + half - x);
    cairo_set_font_size(c, fit_size(c, s, size, room));
    cairo_text_extents(c, s, &e);
    cairo_move_to(c, x - e.x_advance * align, y);
    cairo_set_source_rgba(c, 0, 0, 0, a);
    cairo_show_text(c, s);
}

/* Text bent along the rim, centred on angle mid (radians, clockwise from 3 o'clock), upright at the top */
static void hud_arc_text(cairo_t *c, const char *s, double r, double mid, double size, double a)
{
    cairo_text_extents_t e;
    char ch[2] = { 0, 0 };
    cairo_select_font_face(c, FONT, CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(c, size);
    cairo_text_extents(c, s, &e);
    double ang = mid - e.x_advance / r / 2;
    cairo_set_source_rgba(c, 0, 0, 0, a);
    for (const char *p = s; *p; p++) {
        ch[0] = *p;
        cairo_text_extents(c, ch, &e);
        double half = e.x_advance / 2 / r;
        cairo_save(c);
        cairo_translate(c, CX + cos(ang + half) * r, CY + sin(ang + half) * r);
        cairo_rotate(c, ang + half + M_PI / 2);
        cairo_move_to(c, -e.x_advance / 2, 0);
        cairo_show_text(c, ch);
        cairo_restore(c);
        ang += 2 * half;
    }
}

/* where text can be: x, y, w, h */
static const int hud_box[][4] = { { 16, 14, 448, 154 }, { 110, 438, 260, 38 } };
#define N_HUD_BOX 2

static void update_hud(const shown_t *sh, double t)
{
    static int cpu_i = -1, temp_i = -1, ram_i = -1, w_i = -1, load_i = -1;
    char key[160], cpu_s[16], temp_s[16], ram_s[24], age_s[24], pw_s[24];

    if (t < hud_next)
        return;
    hud_next = t + 0.25;
    /* hysteresis: a number only changes once the value is clearly past it */
    if (cpu_i < 0 || fabs(sh->cpu * 100 - cpu_i) > 0.8)
        cpu_i = (int)lround(sh->cpu * 100);
    if (temp_i < 0 || fabs(sh->tctl - temp_i) > 0.8)
        temp_i = (int)lround(sh->tctl);
    if (ram_i < 0 || fabs(sh->ram_gb - ram_i) > 0.8)
        ram_i = (int)lround(sh->ram_gb);
    snprintf(cpu_s, sizeof(cpu_s), "%d%%", cpu_i);
    snprintf(temp_s, sizeof(temp_s), "%d°", temp_i);
    snprintf(ram_s, sizeof(ram_s), "RAM %d GB", ram_i);
    /* the patient's age is the uptime; metabolism is CPU package watts (or load average without root) */
    int up_m = (int)(sh->uptime / 60);
    if (up_m >= 1440)
        snprintf(age_s, sizeof(age_s), "AGE %dd %dh", up_m / 1440, up_m / 60 % 24);
    else if (up_m >= 60)
        snprintf(age_s, sizeof(age_s), "AGE %dh %dm", up_m / 60, up_m % 60);
    else
        snprintf(age_s, sizeof(age_s), "AGE %dm", up_m);
    if (sh->pkg_w >= 0) {
        if (w_i < 0 || fabs(sh->pkg_w - w_i) > 1.5)
            w_i = (int)lround(sh->pkg_w);
        snprintf(pw_s, sizeof(pw_s), "CPU %d W", w_i);
    } else {
        if (load_i < 0 || fabs(sh->load1 * 10 - load_i) > 0.8)
            load_i = (int)lround(sh->load1 * 10);
        snprintf(pw_s, sizeof(pw_s), "LOAD %d.%d", load_i / 10, load_i % 10);
    }
    snprintf(key, sizeof(key), "%s|%s|%s|%s|%s", cpu_s, temp_s, ram_s, age_s, pw_s);
    if (!strcmp(key, hud_key))
        return;
    strcpy(hud_key, key);

    cairo_t *c = cairo_create(hud_mask);
    cairo_set_operator(c, CAIRO_OPERATOR_CLEAR);
    cairo_paint(c);
    cairo_set_operator(c, CAIRO_OPERATOR_OVER);
    hud_text(c, "CPU", 190, 108, 22, 0, 1, 0.7);
    hud_text(c, cpu_s, 192, 152, 44, 1, 1, 1);
    hud_text(c, "TEMP", 290, 108, 22, 0, 0, 0.7);
    hud_text(c, temp_s, 288, 152, 44, 1, 0, 1);
    hud_text(c, ram_s, CX, 466, 22, 1, 0.5, 0.9);
    hud_arc_text(c, age_s, ARC_TEXT_R, -M_PI * 0.625, 22, 0.8);
    hud_arc_text(c, pw_s, ARC_TEXT_R, -M_PI * 0.375, 22, 0.8);
    cairo_destroy(c);

    /* dark halo so the text reads over bones */
    cairo_surface_flush(hud_mask);
    cairo_surface_flush(hud_shadow);
    memcpy(cairo_image_surface_get_data(hud_shadow), cairo_image_surface_get_data(hud_mask),
           cairo_image_surface_get_stride(hud_mask) * SIZE);
    unsigned char *d = cairo_image_surface_get_data(hud_shadow);
    int st = cairo_image_surface_get_stride(hud_shadow);
    for (int b = 0; b < N_HUD_BOX; b++) {
        const int *r = hud_box[b];
        blur_a8(d + r[1] * st + r[0], r[2], r[3], st, 3);
        for (int y = r[1]; y < r[1] + r[3]; y++)
            for (int x = r[0]; x < r[0] + r[2]; x++)
                d[y * st + x] = (unsigned char)(d[y * st + x] * 3 > 255 ? 255 : d[y * st + x] * 3);
    }
    cairo_surface_mark_dirty(hud_shadow);
}

static void render(cairo_t *cr, const stats *g, const shown_t *sh, double t)
{
    (void)g;
    rgb tint = scan_tint(sh->heat), hc = heart_tint(sh->heat);

    cairo_set_operator(cr, CAIRO_OPERATOR_SOURCE);
    cairo_set_source_surface(cr, bg_surf, 0, 0);
    cairo_paint(cr);

    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    /* fever: the rim smoulders */
    double fever = smoothstep(0.7, 1.0, sh->heat);
    if (fever > 0.01) {
        set_rgba(cr, (rgb){ hc.r, hc.g * 0.5, hc.b * 0.3 }, fever * (0.72 + 0.28 * sin(t * 2.5)));
        cairo_mask_surface(cr, rim_mask, 0, 0);
    }

    set_rgba(cr, tint, 1);
    cairo_mask_surface(cr, fig_mask, 0, 0);
    set_rgba(cr, pale(tint, 0.55), 0.55);
    cairo_mask_surface(cr, core_mask, 0, 0);

    draw_lungs(cr, sh, tint, t);
    draw_gut(cr, sh, tint);
    draw_blood(cr, sh, tint, hc);
    draw_spine(cr, sh, tint, t);
    draw_nerves(cr, tint);
    draw_heart(cr, sh, hc);
    draw_brain(cr, sh, tint, t);
    draw_scanline(cr, tint, t);
    draw_ecg(cr, hc, t);
    draw_gauges(cr, sh, tint, hc);

    update_hud(sh, t);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    for (int b = 0; b < N_HUD_BOX; b++)
        cairo_rectangle(cr, hud_box[b][0], hud_box[b][1], hud_box[b][2], hud_box[b][3]);
    cairo_clip(cr);
    cairo_set_source_rgba(cr, 0, 0, 0, 0.75);
    cairo_mask_surface(cr, hud_shadow, 0, 0);
    rgb tc = pale(tint, 0.65);
    set_rgba(cr, tc, 1);
    cairo_mask_surface(cr, hud_mask, 0, 0);
    cairo_reset_clip(cr);
}

/* Ease the shown values toward the readings */
static void ease(shown_t *sh, const stats *g, const sys_stats *s, double dt)
{
    double k = fmin(1, dt * 3), kt = fmin(1, dt * 1.2);
    sh->cpu += (s->cpu - sh->cpu) * k;
    for (int i = 0; i < MAX_THREADS; i++)
        sh->thread[i] += (s->thread[i] - sh->thread[i]) * fmin(1, dt * 5);
    if (s->tctl > 1) {
        if (sh->tctl < 1)
            sh->tctl = s->tctl;
        sh->tctl += (s->tctl - sh->tctl) * kt;
    }
    sh->heat += (clamp01((sh->tctl - 45) / 47) - sh->heat) * kt;
    double ram = s->ram_total > 0 ? s->ram_used / s->ram_total : 0;
    sh->ram += (ram - sh->ram) * k;
    sh->ram_gb += (s->ram_used - sh->ram_gb) * k;
    for (int i = 0; i < 2; i++)
        sh->gpu[i] += (gpu_act(g, i) - sh->gpu[i]) * k;
    sh->rd += (act(s->disk_rd, DISK_FULL) - sh->rd) * k;
    sh->wr += (act(s->disk_wr, DISK_FULL) - sh->wr) * k;
    sh->rx += (act(s->net_rx, NET_FULL) - sh->rx) * k;
    sh->tx += (act(s->net_tx, NET_FULL) - sh->tx) * k;
    sh->intr += (clamp01(log1p(s->intr_s / 2e4) / log1p(1e6 / 2e4)) - sh->intr) * k;
    sh->pf += (clamp01(log1p(s->pgfault_s / 1e4) / log1p(2e6 / 1e4)) - sh->pf) * k;
    sh->pfmaj += (clamp01(log1p(s->pgmajfault_s / 10) / log1p(5e4 / 10)) - sh->pfmaj) * k;
    double psi = fmax(s->psi_cpu, fmax(s->psi_mem, s->psi_io));
    sh->psi += (pow(clamp01(psi / 30), 0.7) - sh->psi) * fmin(1, dt * 1.5);
    double up = 0, dn = 0;
    for (int i = 0; i < N_GPUS; i++)
        up += s->pcie_rx[i], dn += s->pcie_tx[i];
    sh->pcie_up += (clamp01(log1p(up / 5e7) / log1p(3e10 / 5e7)) - sh->pcie_up) * k;
    sh->pcie_dn += (clamp01(log1p(dn / 5e7) / log1p(3e10 / 5e7)) - sh->pcie_dn) * k;
    sh->pkg_w = s->pkg_w < 0 ? -1 : sh->pkg_w + (s->pkg_w - fmax(0, sh->pkg_w)) * k;
    sh->load1 += (s->load1 - sh->load1) * k;
    sh->uptime = s->uptime;
}

/* ---------------------------------------------------------------- main */

int main(int argc, char **argv)
{
    cairo_surface_t *surf;
    cairo_t *cr;
    tjhandle tj;
    unsigned char *jpeg = NULL;
    stats g = { 0 };
    sys_stats s = { 0 };
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
        else if (!strcmp(argv[i], "--showcase"))
            showcase = demo = 1;
        else if (!strcmp(argv[i], "--bench"))
            bench = 1;
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);
    build_static();

    surf = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cr = cairo_create(surf);
    tj = tj3Init(TJINIT_COMPRESS);
    tj3Set(tj, TJPARAM_QUALITY, JPEG_QUALITY);
    tj3Set(tj, TJPARAM_SUBSAMP, TJSAMP_420);

    if (bench) {
        /* showcase moments: maxed fever, mid, asleep */
        struct { double at; const char *png; } scenes[] = {
            { 22.0, "xray_preview.png" }, { 13.0, "xray_mid.png" }, { 1.0, "xray_idle.png" },
        };
        for (int k = 0; k < 3; k++) {
            srand(5 + k);
            memset(&sh, 0, sizeof(sh));
            memset(sparks, 0, sizeof(sparks));
            beat_n = 0, beat_phase = 0, hud_key[0] = 0, hud_next = 0, blip_n = 0;
            for (int i = 0; i < N_TWITCH; i++)
                twitches[i].age = 1;
            size_t len = 0;
            double b0 = 0;
            int n = 8 * FPS_BUSY;
            for (int i = 0; i < n; i++) {
                double t = scenes[k].at - 8 + i / (double)FPS_BUSY;
                if (i == n - 60)
                    b0 = now_s();
                showcase_poll(&g, &s, t);
                ease(&sh, &g, &s, 1.0 / FPS_BUSY);
                simulate(&g, &sh, 1.0 / FPS_BUSY, t);
                render(cr, &g, &sh, t);
                cairo_surface_flush(surf);
                tj3Compress8(tj, cairo_image_surface_get_data(surf), SIZE, cairo_image_surface_get_stride(surf),
                             SIZE, TJPF_BGRX, &jpeg, &len);
            }
            printf("%s: %.2f ms/frame, jpeg %zu bytes\n", scenes[k].png, (now_s() - b0) * 1000 / 60, len);
            cairo_surface_write_to_png(surf, scenes[k].png);
        }
        return 0;
    }

    if (!demo) {
        gpus_init();
        sys_init();
    }

    while ((fd = lcd_open()) < 0 && !stop)
        sleep(2);
    if (fd >= 0)
        lcd_brightness(fd, 100);

    t0 = last = now_s();
    while (!stop) {
        double t = now_s(), dt = t - last;
        last = t;
        if (dt > 0.25)
            dt = 0.25;

        if (showcase) {
            showcase_poll(&g, &s, t - t0);
        } else if (t >= next_poll) {
            next_poll = t + 0.5;
            if (demo) {
                /* fresh values each poll from the clock; nothing is scaled in place */
                static stats demo_base;
                demo_poll(&demo_base, t - t0);
                g = demo_base;
                g.power[0] = 30 + 520 * g.load[0];
                g.power[1] = 34 + 510 * g.load[1];
                sys_demo(&s, t - t0);
            } else {
                gpus_poll(&g);
                if (gpu_source)
                    gpu_rate_poll(&g);
                else
                    vllm_poll(&g, t);
                sys_poll(&s, t);
            }
        }

        ease(&sh, &g, &s, dt);
        simulate(&g, &sh, dt, t - t0);
        render(cr, &g, &sh, t - t0);
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

        /* a resting machine gets the idle frame rate */
        int idle = sh.cpu < 0.06 && sh.gpu[0] < 0.05 && sh.gpu[1] < 0.05 && sh.rd < 0.4 && sh.wr < 0.4 &&
                   sh.rx < 0.4 && !showcase;
        double spare = 1.0 / (idle ? FPS_IDLE : FPS_BUSY) - (now_s() - t);
        if (spare > 0) {
            struct timespec ts = { 0, (long)(spare * 1e9) };
            nanosleep(&ts, NULL);
        }
    }

    tj3Free(jpeg);
    tj3Destroy(tj);
    cairo_destroy(cr);
    cairo_surface_destroy(surf);
    cairo_surface_destroy(bg_surf);
    cairo_surface_destroy(fig_mask);
    cairo_surface_destroy(core_mask);
    cairo_surface_destroy(lung_fill_mask);
    cairo_surface_destroy(heart_spr);
    cairo_surface_destroy(brain_spr);
    cairo_surface_destroy(rim_mask);
    cairo_surface_destroy(hud_mask);
    cairo_surface_destroy(hud_shadow);
    for (int i = 0; i < 2; i++)
        cairo_path_destroy(lung_path[i]);
    cairo_path_destroy(stom_path);
    if (nvml_ok)
        nvmlShutdown();
    if (fd >= 0)
        close(fd);
    return 0;
}
