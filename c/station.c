/*
 * station: a slowly turning ring space station above a planet, on the iCUE LINK AIO
 * pump LCD, driven by this machine's own sensors.
 *
 * The ring carries 32 habitat pods, one per CPU thread, whose windows light up with
 * that thread's load. A cargo bay fills with containers as RAM is used, three docking
 * ports (one per NVMe drive) see shuttles arrive for writes and leave for reads, a
 * comms dish on the mast beams packets to a relay satellite for network traffic, and
 * two reactor cores on the spindle (the two GPUs) glow with GPU power. Pods and docks
 * vent puffs of coolant as CPU and NVMe temperatures climb. Escape capsules launch
 * with new processes, visiting ships wait in a holding orbit (open TCP connections),
 * solar wings light up with CPU package watts, radiators unfold with GPU fan speed,
 * red strobes warn of CPU, I/O or memory pressure (PSI), and uptime is the mission day.
 * The planet and starfield are a painting in assets/station/; the station is drawn
 * with cairo from layers and sprites cached at startup.
 * Run with --demo to simulate data, --showcase for a scripted 36 s arc,
 * --bench to write preview PNGs.
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
 * Everything the station shows besides the GPUs, read from /proc and /sys. hwmon
 * chips are found by name (never by hwmonN), NVMe sensors are matched to their drive
 * through the hwmon's device link. Rates are differenced between polls; call
 * sys_poll at most a couple of times a second, it only reads small text files.
 */
#define SYS_MAX_CPU     64
#define SYS_N_NVME      3
#define SYS_N_DIMM      2
#define SYS_N_BOARD     5

static const char *sys_nvme_dev[SYS_N_NVME] = { "nvme0n1", "nvme1n1", "nvme2n1" };
static const char *sys_net_if[]             = { "enp12s0", "tailscale0" };
#define SYS_N_NET   (int)(sizeof(sys_net_if) / sizeof(sys_net_if[0]))
static const char *sys_board_label[SYS_N_BOARD] = { "CPU", "CPU Package", "Motherboard", "T_Sensor", "VRM" };

typedef struct {
    int    n_cpu;                           /* threads seen in /proc/stat */
    double cpu[SYS_MAX_CPU];                /* per-thread load 0..1 */
    double cpu_total;                       /* 0..1 */
    double cpu_temp;                        /* k10temp Tctl, C */
    double board_temp[SYS_N_BOARD];         /* asusec, in sys_board_label order, C */
    double fan_rpm;                         /* asusec fan1 */
    double mem_total, mem_used, mem_cached; /* GB; used = total - available */
    double dimm_temp[SYS_N_DIMM];           /* spd5118, C */
    double nvme_temp[SYS_N_NVME];           /* C */
    double nvme_rd[SYS_N_NVME], nvme_wr[SYS_N_NVME];   /* bytes/s */
    double net_rx, net_tx;                  /* bytes/s, summed over sys_net_if */
    double forks, ctxt;                     /* new processes and context switches per second */
    double psi_cpu, psi_io, psi_mem;        /* /proc/pressure "some avg10", % of time stalled */
    double tcp;                             /* open TCP connections (sockstat inuse) */
    double uptime;                          /* s */
    double cpu_watts;                       /* RAPL package power; NAN when not readable (needs root) */
    double gpu_fan[2];                      /* % (NVML), filled by gpu_fans_poll */
} sys_stats;

static char sys_k10[300], sys_board[SYS_N_BOARD][300], sys_fan[300];
static char sys_dimm[SYS_N_DIMM][300], sys_nvme_hw[SYS_N_NVME][300];

static int read_file(const char *path, char *buf, int cap)
{
    int fd = open(path, O_RDONLY | O_CLOEXEC), n;
    if (fd < 0)
        return -1;
    n = (int)read(fd, buf, cap - 1);
    close(fd);
    if (n < 0)
        return -1;
    buf[n] = 0;
    return n;
}

/* millidegrees (or rpm) from a sysfs file; NAN if it's missing */
static double read_milli(const char *path, double scale)
{
    char b[32];
    if (!*path || read_file(path, b, sizeof(b)) <= 0)
        return NAN;
    return strtod(b, NULL) * scale;
}

static void sys_init(void)
{
    DIR *d = opendir("/sys/class/hwmon");
    struct dirent *e;
    char p[512], name[64], lbl[64], link[512];
    int ndimm = 0;

    if (!d)
        return;
    while ((e = readdir(d))) {
        if (strncmp(e->d_name, "hwmon", 5))
            continue;
        snprintf(p, sizeof(p), "/sys/class/hwmon/%s/name", e->d_name);
        if (read_file(p, name, sizeof(name)) <= 0)
            continue;
        name[strcspn(name, "\n")] = 0;
        if (!strcmp(name, "k10temp")) {
            snprintf(sys_k10, sizeof(sys_k10), "/sys/class/hwmon/%s/temp1_input", e->d_name);
        } else if (!strcmp(name, "asusec")) {
            for (int i = 1; i <= 8; i++) {
                snprintf(p, sizeof(p), "/sys/class/hwmon/%s/temp%d_label", e->d_name, i);
                if (read_file(p, lbl, sizeof(lbl)) <= 0)
                    continue;
                lbl[strcspn(lbl, "\n")] = 0;
                for (int k = 0; k < SYS_N_BOARD; k++)
                    if (!strcmp(lbl, sys_board_label[k]))
                        snprintf(sys_board[k], sizeof(sys_board[k]), "/sys/class/hwmon/%s/temp%d_input", e->d_name, i);
            }
            snprintf(sys_fan, sizeof(sys_fan), "/sys/class/hwmon/%s/fan1_input", e->d_name);
        } else if (!strcmp(name, "spd5118") && ndimm < SYS_N_DIMM) {
            snprintf(sys_dimm[ndimm++], sizeof(sys_dimm[0]), "/sys/class/hwmon/%s/temp1_input", e->d_name);
        } else if (!strcmp(name, "nvme")) {
            /* the device link ends in .../nvme/nvmeN, which is the controller of nvmeNn1 */
            snprintf(p, sizeof(p), "/sys/class/hwmon/%s/device", e->d_name);
            ssize_t n = readlink(p, link, sizeof(link) - 1);
            if (n <= 0)
                continue;
            link[n] = 0;
            const char *base = strrchr(link, '/');
            base = base ? base + 1 : link;
            for (int k = 0; k < SYS_N_NVME; k++) {
                size_t bl = strlen(base);
                if (!strncmp(sys_nvme_dev[k], base, bl) && sys_nvme_dev[k][bl] == 'n')
                    snprintf(sys_nvme_hw[k], sizeof(sys_nvme_hw[k]), "/sys/class/hwmon/%s/temp1_input", e->d_name);
            }
        }
    }
    closedir(d);
}

static void sys_poll(sys_stats *y, double t)
{
    static char buf[1 << 16];
    static unsigned long long prev_busy[SYS_MAX_CPU + 1], prev_all[SYS_MAX_CPU + 1];
    static unsigned long long prev_rd[SYS_N_NVME], prev_wr[SYS_N_NVME], prev_rx, prev_tx, prev_forks, prev_ctxt;
    static unsigned long long prev_uj, max_uj;
    static int rapl = 1;                    /* cleared for good once it can't be read */
    static double last_t;
    static int have;
    double dt = t - last_t;
    char *line, *save;

    /* per-thread load from /proc/stat */
    if (read_file("/proc/stat", buf, sizeof(buf)) > 0) {
        int n = 0;
        for (line = strtok_r(buf, "\n", &save); line; line = strtok_r(NULL, "\n", &save)) {
            unsigned long long v[10] = { 0 };
            int idx;
            if (!strncmp(line, "processes ", 10) || !strncmp(line, "ctxt ", 5)) {
                int fk = line[0] == 'p';
                unsigned long long v1 = strtoull(line + (fk ? 10 : 5), NULL, 10);
                unsigned long long *prev = fk ? &prev_forks : &prev_ctxt;
                double rate = have && dt > 0 && v1 >= *prev ? (v1 - *prev) / dt : 0;
                if (fk)
                    y->forks = rate;
                else
                    y->ctxt = rate;
                *prev = v1;
                continue;
            }
            if (strncmp(line, "cpu", 3))
                continue;
            char *p = line + 3;
            idx = *p == ' ' ? SYS_MAX_CPU : (int)strtol(p, &p, 10);
            if (idx > SYS_MAX_CPU)
                continue;
            for (int k = 0; k < 10; k++)
                v[k] = strtoull(p, &p, 10);
            unsigned long long all = 0, idle = v[3] + v[4];
            for (int k = 0; k < 8; k++)             /* guest time is already in user */
                all += v[k];
            unsigned long long busy = all - idle;
            double load = 0;
            if (have && all > prev_all[idx])
                load = (double)(busy - prev_busy[idx]) / (double)(all - prev_all[idx]);
            prev_busy[idx] = busy;
            prev_all[idx] = all;
            if (idx == SYS_MAX_CPU)
                y->cpu_total = clamp01(load);
            else {
                y->cpu[idx] = clamp01(load);
                if (idx + 1 > n)
                    n = idx + 1;
            }
        }
        y->n_cpu = n;
    }

    /* memory */
    if (read_file("/proc/meminfo", buf, sizeof(buf)) > 0) {
        double tot = 0, avail = 0, cached = 0;
        char *p;
        if ((p = strstr(buf, "MemTotal:")))
            tot = strtod(p + 9, NULL);
        if ((p = strstr(buf, "MemAvailable:")))
            avail = strtod(p + 13, NULL);
        if ((p = strstr(buf, "\nCached:")))
            cached = strtod(p + 8, NULL);
        y->mem_total = tot / 1048576.0;
        y->mem_used = (tot - avail) / 1048576.0;
        y->mem_cached = cached / 1048576.0;
    }

    /* NVMe throughput: sectors read (field 6) and written (field 10), 512 bytes each */
    if (read_file("/proc/diskstats", buf, sizeof(buf)) > 0) {
        for (int k = 0; k < SYS_N_NVME; k++) {
            char pat[32];
            snprintf(pat, sizeof(pat), " %s ", sys_nvme_dev[k]);
            char *p = strstr(buf, pat);
            if (!p)
                continue;
            unsigned long long f[8];
            p += strlen(pat);
            for (int i = 0; i < 8; i++)
                f[i] = strtoull(p, &p, 10);
            unsigned long long rd = f[2] * 512, wr = f[6] * 512;
            if (have && dt > 0) {
                y->nvme_rd[k] = rd >= prev_rd[k] ? (rd - prev_rd[k]) / dt : 0;
                y->nvme_wr[k] = wr >= prev_wr[k] ? (wr - prev_wr[k]) / dt : 0;
            }
            prev_rd[k] = rd;
            prev_wr[k] = wr;
        }
    }

    /* network */
    if (read_file("/proc/net/dev", buf, sizeof(buf)) > 0) {
        unsigned long long rx = 0, tx = 0;
        for (int k = 0; k < SYS_N_NET; k++) {
            char pat[32];
            snprintf(pat, sizeof(pat), "%s:", sys_net_if[k]);
            char *p = strstr(buf, pat);
            if (!p)
                continue;
            unsigned long long f[9];
            p += strlen(pat);
            for (int i = 0; i < 9; i++)
                f[i] = strtoull(p, &p, 10);
            rx += f[0];
            tx += f[8];
        }
        if (have && dt > 0) {
            y->net_rx = rx >= prev_rx ? (rx - prev_rx) / dt : 0;
            y->net_tx = tx >= prev_tx ? (tx - prev_tx) / dt : 0;
        }
        prev_rx = rx;
        prev_tx = tx;
    }

    /* pressure stall info */
    static const char *psi_file[3] = { "/proc/pressure/cpu", "/proc/pressure/io", "/proc/pressure/memory" };
    double *psi_out[3] = { &y->psi_cpu, &y->psi_io, &y->psi_mem };
    for (int k = 0; k < 3; k++) {
        char pb[256], *p;
        if (read_file(psi_file[k], pb, sizeof(pb)) > 0 && (p = strstr(pb, "some avg10=")))
            *psi_out[k] = strtod(p + 11, NULL);
    }

    /* open TCP connections and uptime */
    if (read_file("/proc/net/sockstat", buf, sizeof(buf)) > 0) {
        char *p = strstr(buf, "TCP: inuse ");
        if (p)
            y->tcp = strtod(p + 11, NULL);
    }
    if (read_file("/proc/uptime", buf, sizeof(buf)) > 0)
        y->uptime = strtod(buf, NULL);

    /* CPU package watts from RAPL (root only; without it the solar array follows load) */
    y->cpu_watts = NAN;
    if (rapl) {
        char eb[32];
        if (!max_uj && read_file("/sys/class/powercap/intel-rapl:0/max_energy_range_uj", eb, sizeof(eb)) > 0)
            max_uj = strtoull(eb, NULL, 10);
        if (read_file("/sys/class/powercap/intel-rapl:0/energy_uj", eb, sizeof(eb)) > 0) {
            unsigned long long uj = strtoull(eb, NULL, 10);
            if (have && prev_uj && dt > 0) {
                unsigned long long d = uj >= prev_uj ? uj - prev_uj : uj + max_uj - prev_uj;
                y->cpu_watts = d / 1e6 / dt;
            }
            prev_uj = uj;
        } else {
            rapl = 0;
        }
    }

    /* temperatures */
    y->cpu_temp = read_milli(sys_k10, 0.001);
    for (int k = 0; k < SYS_N_BOARD; k++)
        y->board_temp[k] = read_milli(sys_board[k], 0.001);
    y->fan_rpm = read_milli(sys_fan, 1);
    for (int k = 0; k < SYS_N_DIMM; k++)
        y->dimm_temp[k] = read_milli(sys_dimm[k], 0.001);
    for (int k = 0; k < SYS_N_NVME; k++)
        y->nvme_temp[k] = read_milli(sys_nvme_hw[k], 0.001);

    last_t = t;
    have = 1;
}

/* GPU fan % from NVML, for the radiators (uses the handles from gpus_init) */
static void gpu_fans_poll(sys_stats *y)
{
    for (int i = 0; i < N_GPUS && i < 2; i++) {
        unsigned int pct;
        if (nvml_ok && nvml_dev[i] && nvmlDeviceGetFanSpeed(nvml_dev[i], &pct) == NVML_SUCCESS)
            y->gpu_fan[i] = pct;
    }
}

/* ---- end of system sensors ---- */

/* ---------------------------------------------------------------- scene */

/*
 * The station lives in a small 3D world seen with an orthographic camera tilted
 * down: X right, Y up the spin axis, Z toward the viewer. screen = (X, Z*TS - Y*TC).
 * The ring, hub and mast are symmetric about the axis, so they are drawn once into
 * cached layers and only the things on them (pods, containers, docks, spokes,
 * seams and windows) are drawn each frame at the current rotation. Draw order is
 * back half, spindle and hub, front half, then everything in open space.
 */
#define FPS_BUSY        20
#define FPS_IDLE        15
#define FPS_SHOW        24
#define CX              240.0
#define CY              222.0
#define TS              0.6         /* sin of the camera tilt */
#define TC              0.8
#define R_IN            165.0       /* ring deck, inner and outer radius */
#define R_OUT           191.0
#define RING_H          16.0        /* wall height below the deck */
#define HUB_R           32.0
#define HUB_Y0          -10.0
#define HUB_Y1          8.0
#define MAST_R          4.5
#define MAST_LO         -98.0
#define MAST_HI         110.0
#define REACT_Y0        64.0        /* GPU 0's core, above the hub */
#define REACT_Y1        -62.0       /* GPU 1's core, below */
#define CORE_R          10.5
#define CONT_R          21.0        /* containment ring around a core */
#define DISH_Y          112.0
#define SOLAR_Y         92.0        /* solar wings on the upper mast */
#define SOLAR_IN        12.0
#define SOLAR_OUT       112.0
#define SOLAR_W         9.0         /* half width across the wing */
#define RAD_LEN         26.0        /* radiator panel length */
#define ORBIT_R         224.0       /* holding orbit of the visiting ships */
#define ORBIT_Y         34.0
#define N_SHIPS         40
#define MAX_CAPS        48
#define SAT_X           378.0       /* relay satellite, in screen space */
#define SAT_Y           100.0
#define ROT_PERIOD      140.0       /* seconds per turn of the ring */
#define N_PODS          32
#define BAY_COLS        14
#define BAY_LANES       3
#define N_SLOTS         (BAY_COLS * BAY_LANES)
#define BAY_HALF        (30.0 * M_PI / 180)
#define MAX_SHUTTLES    48
#define MAX_PUFFS       160
#define MAX_PACKETS     64
#define N_TWINKLE       34
#define HUD_Y           360
#define HUD_H           110
#define N_PAGES         4
#define PAGE_SECS       7.0

static double DEG(double d) { return d * M_PI / 180; }

static const rgb GPU_COL[2] = { { 0.30, 0.66, 1.00 }, { 1.00, 0.52, 0.18 } };
static const rgb HULL     = { 0.66, 0.70, 0.76 };
static const rgb HULL_DK  = { 0.34, 0.37, 0.42 };
static const rgb LAMP     = { 1.00, 0.78, 0.44 };      /* warm habitat light */
static const rgb LAMP_HOT = { 1.00, 0.96, 0.88 };
static const rgb CYAN     = { 0.45, 0.88, 1.00 };
static const rgb AMBER    = { 1.00, 0.72, 0.34 };

/* sun up-left and a little behind, planet bounce from below and in front */
static const double LX = -0.553, LY = 0.553, LZ = -0.623;
static const double PX = 0.15, PY = -0.45, PZ = 0.88;

static rgb shade(rgb c, double nx, double ny, double nz)
{
    double sun = fmax(0, nx * LX + ny * LY + nz * LZ);
    double pl = fmax(0, (nx * PX + ny * PY + nz * PZ) / 1.0118);
    double amb = 0.24 + 0.14 * fmax(0, ny);
    rgb o = { c.r * (amb + 0.80 * sun) + 0.06 * pl, c.g * (amb + 0.78 * sun) + 0.13 * pl,
              c.b * (amb + 0.72 * sun) + 0.24 * pl };
    return o;
}

typedef struct {
    int    alive, dock, arrive;
    double age, dur, hold, ang, y0, r0;     /* spawn direction (inertial) and height */
    double lx, ly;                          /* last drawn position, for the trail */
} shuttle;

typedef struct { int alive; double x, y, z, vx, vy, vz, age, life, size; } puff;
typedef puff capsule;                       /* same motion, drawn as a bright dart */
typedef struct { int alive, up; double u, speed; } packet;
typedef struct { double x, y, r, ph, sp; } star;

/* values as drawn: eased toward the latest poll every frame */
typedef struct {
    double pod[N_PODS];
    double cpu, cpu_temp, mem_used, mem_cached, mem_total;
    double rd[SYS_N_NVME], wr[SYS_N_NVME], nvme_temp[SYS_N_NVME];
    double rx, tx;
    double power[N_GPUS], load[N_GPUS], temp[N_GPUS], tok, fan[N_GPUS];
    double forks, tcp, psi_cpu, psi_io, psi_mem, cpu_watts, uptime;
    int    watts_real;                      /* cpu_watts is measured, not estimated */
} shown_t;

static cairo_surface_t *bg_layer, *wall_layer, *mast_lo_layer, *hub_layer, *mast_hi_layer;
static cairo_surface_t *solar_lit, *pod_glow, *streak_spr[2], *core_dark[2], *core_hot[2], *react_glow[2];
static cairo_surface_t *puff_spr, *dot_spr[4], *sat_spr, *hud_cache[N_PAGES];
static shuttle shuttles[MAX_SHUTTLES];
static puff    puffs[MAX_PUFFS];
static capsule caps[MAX_CAPS];
static double  ship_vis[N_SHIPS], ship_r[N_SHIPS], ship_y[N_SHIPS], orbit_rot, spawn_cap;
static int     ship_order[N_SHIPS], n_ships;
static packet  packets[MAX_PACKETS];
static star    twinkle[N_TWINKLE];
static double  pod_ang[N_PODS], slot_fill[N_SLOTS], slot_ghost[N_SLOTS];
static int     slot_ci[N_SLOTS];
static double  rot, spawn_rd[SYS_N_NVME], spawn_wr[SYS_N_NVME], spawn_vent_cpu, spawn_vent_nvme[SYS_N_NVME];
static double  spawn_rx, spawn_tx, core_phase[N_GPUS], spark_phase[N_GPUS];
static int     n_used_slots, n_cache_slots;
static char    hud_key[N_PAGES][160];

enum { DOT_CYAN, DOT_AMBER, DOT_GREEN, DOT_RED };

static double frand(void) { return rand() / (double)RAND_MAX; }
static double smooth01(double x) { x = clamp01(x); return x * x * (3 - 2 * x); }

static void proj(double X, double Y, double Z, double *sx, double *sy)
{
    *sx = CX + X;
    *sy = CY + Z * TS - Y * TC;
}

static void polar(double r, double th, double Y, double *sx, double *sy)
{
    proj(r * cos(th), Y, r * sin(th), sx, sy);
}

static void quad(cairo_t *cr, const double *x, const double *y)
{
    cairo_move_to(cr, x[0], y[0]);
    for (int i = 1; i < 4; i++)
        cairo_line_to(cr, x[i], y[i]);
    cairo_close_path(cr);
}

static cairo_surface_t *load_asset(const char *name)
{
    char exe[512], path[1024];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n > 0) {
        exe[n] = 0;
        char *slash = strrchr(exe, '/');
        if (slash)
            *slash = 0;
        snprintf(path, sizeof(path), "%s/assets/station/%s", exe, name);
        cairo_surface_t *s = cairo_image_surface_create_from_png(path);
        if (cairo_surface_status(s) == CAIRO_STATUS_SUCCESS)
            return s;
        cairo_surface_destroy(s);
    }
    snprintf(path, sizeof(path), "assets/station/%s", name);
    cairo_surface_t *s = cairo_image_surface_create_from_png(path);
    if (cairo_surface_status(s) != CAIRO_STATUS_SUCCESS) {
        fprintf(stderr, "station: can't load %s (looked next to the binary and in ./assets/station)\n", name);
        exit(1);
    }
    return s;
}

static cairo_surface_t *new_surf(int w, int h, cairo_t **cr)
{
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w, h);
    *cr = cairo_create(s);
    return s;
}

/* A soft round glow of colour c, radius r, peak alpha a */
static cairo_surface_t *make_glow(double r, rgb c, double a, double core)
{
    cairo_t *cr;
    int w = (int)ceil(r * 2) + 2;
    cairo_surface_t *s = new_surf(w, w, &cr);
    cairo_pattern_t *g = cairo_pattern_create_radial(w / 2.0, w / 2.0, 0, w / 2.0, w / 2.0, r);
    cairo_pattern_add_color_stop_rgba(g, 0.00, c.r, c.g, c.b, a);
    cairo_pattern_add_color_stop_rgba(g, core, c.r, c.g, c.b, a * 0.45);
    cairo_pattern_add_color_stop_rgba(g, 0.55, c.r, c.g, c.b, a * 0.12);
    cairo_pattern_add_color_stop_rgba(g, 1.00, c.r, c.g, c.b, 0);
    cairo_set_source(cr, g);
    cairo_paint(cr);
    cairo_pattern_destroy(g);
    cairo_destroy(cr);
    return s;
}

/* Paint sprite centred on (x,y), scaled by k, with the current operator */
static void blit(cairo_t *cr, cairo_surface_t *spr, double x, double y, double k, double alpha)
{
    double w = cairo_image_surface_get_width(spr), h = cairo_image_surface_get_height(spr);
    if (alpha <= 0.004)
        return;
    cairo_save(cr);
    cairo_translate(cr, x, y);
    if (k != 1)
        cairo_scale(cr, k, k);
    cairo_set_source_surface(cr, spr, -w / 2, -h / 2);
    cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_BILINEAR);
    cairo_rectangle(cr, -w / 2, -h / 2, w, h);
    cairo_clip(cr);
    cairo_paint_with_alpha(cr, alpha);
    cairo_restore(cr);
}

static void blit_add(cairo_t *cr, cairo_surface_t *spr, double x, double y, double k, double alpha)
{
    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    blit(cr, spr, x, y, k, alpha);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
}

/* Paint a full-size cached layer, only over the box where it has pixels */
static void paint_layer(cairo_t *cr, cairo_surface_t *s, double x, double y, double w, double h)
{
    cairo_set_source_surface(cr, s, 0, 0);
    cairo_rectangle(cr, x, y, w, h);
    cairo_fill(cr);
}

/* an elliptical arc of the ring plane at height Y, radius r, from a0 to a1 (ccw if dir > 0) */
static void ring_arc(cairo_t *cr, double r, double Y, double a0, double a1, int dir)
{
    cairo_save(cr);
    cairo_translate(cr, CX, CY - Y * TC);
    cairo_scale(cr, 1, TS);
    if (dir > 0)
        cairo_arc(cr, 0, 0, r, a0, a1);
    else
        cairo_arc_negative(cr, 0, 0, r, a0, a1);
    cairo_restore(cr);
}

/* A vertical cylinder on the axis from Y0 to Y1, radius r, lit from the left */
static void cylinder(cairo_t *cr, double r, double Y0, double Y1, rgb c, int cap)
{
    double x0, yb, yt, dummy;
    proj(0, Y0, 0, &dummy, &yb);
    proj(0, Y1, 0, &dummy, &yt);
    x0 = CX - r;
    cairo_pattern_t *g = cairo_pattern_create_linear(x0, 0, CX + r, 0);
    for (int i = 0; i <= 8; i++) {
        double th = M_PI - i * M_PI / 8;            /* left edge to right edge, front side */
        rgb s = shade(c, cos(th), 0, sin(th));
        cairo_pattern_add_color_stop_rgb(g, i / 8.0, s.r, s.g, s.b);
    }
    cairo_new_path(cr);
    cairo_save(cr);
    cairo_translate(cr, CX, yb);
    cairo_scale(cr, 1, TS);
    cairo_arc(cr, 0, 0, r, 0, M_PI);
    cairo_restore(cr);
    cairo_save(cr);
    cairo_translate(cr, CX, yt);
    cairo_scale(cr, 1, TS);
    cairo_arc_negative(cr, 0, 0, r, M_PI, 0);
    cairo_restore(cr);
    cairo_close_path(cr);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    if (cap) {
        rgb s = shade(c, 0, 1, 0);
        cairo_save(cr);
        cairo_translate(cr, CX, yt);
        cairo_scale(cr, 1, TS);
        cairo_arc(cr, 0, 0, r, 0, 2 * M_PI);
        cairo_restore(cr);
        set_rgb(cr, s);
        cairo_fill_preserve(cr);
        cairo_set_source_rgba(cr, 1, 1, 1, 0.10);
        cairo_set_line_width(cr, 0.8);
        cairo_stroke(cr);
    }
}

/* The walls of the ring between angles, radius r, lit per angle: a horizontal gradient */
static cairo_pattern_t *wall_gradient(double r, double a0, double a1, double nsign, rgb c)
{
    cairo_pattern_t *g = cairo_pattern_create_linear(CX - r, 0, CX + r, 0);
    for (int i = 0; i <= 32; i++) {
        double th = a0 + (a1 - a0) * i / 32.0;
        double x = r * cos(th);
        rgb s = shade(c, nsign * cos(th), 0, nsign * sin(th));
        cairo_pattern_add_color_stop_rgb(g, (x + r) / (2 * r), s.r, s.g, s.b);
    }
    return g;
}

static void draw_solar(cairo_t *cr, int lit);

static void build_static(void)
{
    cairo_t *cr;
    cairo_surface_t *src = load_asset("backdrop.png");

    /* backdrop + the parts of the ring that nothing can cover: deck and back inner wall */
    bg_layer = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cr = cairo_create(bg_layer);
    cairo_save(cr);
    cairo_scale(cr, SIZE / (double)cairo_image_surface_get_width(src), SIZE / (double)cairo_image_surface_get_height(src));
    cairo_set_source_surface(cr, src, 0, 0);
    cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_BEST);
    cairo_paint(cr);
    cairo_restore(cr);
    cairo_surface_destroy(src);

    /* darken the planet a touch under the text */
    cairo_pattern_t *g = cairo_pattern_create_linear(0, HUD_Y - 10, 0, SIZE);
    cairo_pattern_add_color_stop_rgba(g, 0, 0.0, 0.01, 0.04, 0);
    cairo_pattern_add_color_stop_rgba(g, 0.35, 0.0, 0.01, 0.04, 0.42);
    cairo_pattern_add_color_stop_rgba(g, 1, 0.0, 0.01, 0.04, 0.55);
    cairo_set_source(cr, g);
    cairo_rectangle(cr, 0, HUD_Y - 10, SIZE, SIZE);
    cairo_fill(cr);
    cairo_pattern_destroy(g);

    /* soft shadow of the ring on space (a faint dark halo reads as depth against the planet) */
    cairo_set_line_width(cr, 34);
    ring_arc(cr, (R_IN + R_OUT) / 2, -RING_H * 0.9, 0, 2 * M_PI, 1);
    cairo_set_source_rgba(cr, 0, 0, 0.02, 0.18);
    cairo_stroke(cr);

    /* inner wall of the back half, facing us */
    cairo_new_path(cr);
    ring_arc(cr, R_IN, 0, M_PI, 2 * M_PI, 1);
    ring_arc(cr, R_IN, -RING_H, 2 * M_PI, M_PI, -1);
    cairo_close_path(cr);
    g = wall_gradient(R_IN, M_PI, 2 * M_PI, -1, HULL_DK);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    cairo_set_line_width(cr, 1.2);                  /* a structural band along it */
    ring_arc(cr, R_IN, -RING_H * 0.62, M_PI, 2 * M_PI, 1);
    cairo_set_source_rgba(cr, 0, 0, 0, 0.35);
    cairo_stroke(cr);

    /* the deck: an annulus with lips and grooves */
    cairo_new_path(cr);
    ring_arc(cr, R_OUT, 0, 0, 2 * M_PI, 1);
    cairo_new_sub_path(cr);
    ring_arc(cr, R_IN, 0, 2 * M_PI, 0, -1);
    cairo_close_path(cr);
    rgb deck = shade(HULL, 0, 1, 0);
    set_rgb(cr, deck);
    cairo_fill_preserve(cr);
    /* sun sheen, brightest on the side toward the sun */
    g = cairo_pattern_create_radial(CX - 150, CY - 110, 10, CX - 150, CY - 110, 330);
    cairo_pattern_add_color_stop_rgba(g, 0, 1.0, 0.95, 0.85, 0.34);
    cairo_pattern_add_color_stop_rgba(g, 0.5, 1.0, 0.95, 0.85, 0.08);
    cairo_pattern_add_color_stop_rgba(g, 1, 0.3, 0.45, 0.8, 0.10);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    double grooves[] = { R_IN + 3.5, R_IN + 9, R_OUT - 9, R_OUT - 3.5 };
    for (int i = 0; i < 4; i++) {
        ring_arc(cr, grooves[i], 0, 0, 2 * M_PI, 1);
        cairo_set_line_width(cr, i == 0 || i == 3 ? 0.9 : 0.6);
        cairo_set_source_rgba(cr, 0.02, 0.03, 0.06, i == 0 || i == 3 ? 0.45 : 0.28);
        cairo_stroke(cr);
    }
    /* rim highlight on the sun side */
    for (int k = 0; k < 2; k++) {
        double r = k ? R_OUT : R_IN;
        g = cairo_pattern_create_linear(CX - r, CY - r * TS, CX + r, CY + r * TS);
        cairo_pattern_add_color_stop_rgba(g, 0, 1, 0.96, 0.88, k ? 0.75 : 0.35);
        cairo_pattern_add_color_stop_rgba(g, 0.45, 1, 0.96, 0.88, 0.10);
        cairo_pattern_add_color_stop_rgba(g, 1, 0.5, 0.7, 1, k ? 0.18 : 0.05);
        ring_arc(cr, r, 0, 0, 2 * M_PI, 1);
        cairo_set_source(cr, g);
        cairo_set_line_width(cr, 1.0);
        cairo_stroke(cr);
        cairo_pattern_destroy(g);
    }
    cairo_destroy(cr);

    /* outer wall of the front half */
    wall_layer = new_surf(SIZE, SIZE, &cr);
    cairo_new_path(cr);
    ring_arc(cr, R_OUT, 0, 0, M_PI, 1);
    ring_arc(cr, R_OUT, -RING_H, M_PI, 0, -1);
    cairo_close_path(cr);
    g = wall_gradient(R_OUT, 0, M_PI, 1, HULL);
    cairo_set_source(cr, g);
    cairo_fill(cr);
    cairo_pattern_destroy(g);
    ring_arc(cr, R_OUT, -RING_H * 0.30, 0, M_PI, 1);   /* window band shadow */
    cairo_set_line_width(cr, 4.5);
    cairo_set_source_rgba(cr, 0.02, 0.03, 0.06, 0.45);
    cairo_stroke(cr);
    ring_arc(cr, R_OUT, -RING_H * 0.78, 0, M_PI, 1);
    cairo_set_line_width(cr, 1.0);
    cairo_set_source_rgba(cr, 0, 0, 0, 0.35);
    cairo_stroke(cr);
    ring_arc(cr, R_OUT, -RING_H + 0.6, 0, M_PI, 1);    /* planet light on the bottom edge */
    cairo_set_line_width(cr, 1.2);
    cairo_set_source_rgba(cr, 0.35, 0.6, 1.0, 0.35);
    cairo_stroke(cr);
    cairo_destroy(cr);

    /* lower mast, lower reactor housing, then hub, then upper mast, housing and dish */
    mast_lo_layer = new_surf(SIZE, SIZE, &cr);
    cylinder(cr, MAST_R * 0.8, MAST_LO, HUB_Y0, HULL_DK, 0);
    cylinder(cr, 3.2, MAST_LO - 12, MAST_LO, HULL_DK, 0);              /* antenna tip */
    cylinder(cr, 8.0, MAST_LO - 2, MAST_LO + 3, HULL, 1);              /* end plate */
    for (int k = 0; k < 2; k++) {
        double y0 = REACT_Y1 + (k ? 12 : -18), y1 = y0 + 6;
        cylinder(cr, 9.5, y0, y1, HULL, 1);                              /* core collars */
    }
    ring_arc(cr, CONT_R, REACT_Y1, M_PI, 2 * M_PI, 1);                  /* containment ring, back */
    cairo_set_line_width(cr, 3.2);
    set_rgb(cr, shade(HULL_DK, 0, 1, 0));
    cairo_stroke(cr);
    cairo_destroy(cr);

    hub_layer = new_surf(SIZE, SIZE, &cr);
    cylinder(cr, HUB_R, HUB_Y0, HUB_Y1, HULL, 1);
    cylinder(cr, HUB_R * 0.62, HUB_Y1, HUB_Y1 + 5, HULL, 1);
    cylinder(cr, HUB_R * 0.34, HUB_Y1 + 5, HUB_Y1 + 8, HULL_DK, 1);
    {   /* windows around the drum */
        for (int i = 0; i < 20; i++) {
            double th = i * 2 * M_PI / 20 + 0.1, sx, sy;
            if (sin(th) < 0.1)
                continue;
            polar(HUB_R, th, (HUB_Y0 + HUB_Y1) / 2, &sx, &sy);
            cairo_rectangle(cr, sx - 1.1, sy - 1.6, 2.2, 3.2);
        }
        set_rgb(cr, lerp(LAMP, LAMP_HOT, 0.3));
        cairo_fill(cr);
        ring_arc(cr, HUB_R, HUB_Y0 + 3, 0, M_PI, 1);
        cairo_set_source_rgba(cr, 0.35, 0.6, 1.0, 0.25);
        cairo_set_line_width(cr, 1);
        cairo_stroke(cr);
    }
    cairo_destroy(cr);

    mast_hi_layer = new_surf(SIZE, SIZE, &cr);
    cylinder(cr, MAST_R * 0.8, HUB_Y1 + 8, MAST_HI, HULL_DK, 0);
    for (int k = 0; k < 2; k++) {
        double y0 = REACT_Y0 + (k ? 12 : -18), y1 = y0 + 6;
        cylinder(cr, 9.5, y0, y1, HULL, 1);
    }
    ring_arc(cr, CONT_R, REACT_Y0, M_PI, 2 * M_PI, 1);
    cairo_set_line_width(cr, 3.2);
    set_rgb(cr, shade(HULL_DK, 0, 1, 0));
    cairo_stroke(cr);
    draw_solar(cr, 0);
    {   /* comms dish on the mast head, aimed at the relay satellite */
        double dx, dy;
        proj(0, DISH_Y, 0, &dx, &dy);
        double ang = atan2(SAT_Y - dy, SAT_X - dx);
        cylinder(cr, 6, MAST_HI - 3, MAST_HI + 1, HULL, 1);
        cairo_save(cr);
        cairo_translate(cr, dx, dy - 2);
        cairo_rotate(cr, ang);
        /* back of the dish, then the bowl facing the satellite */
        cairo_save(cr);
        cairo_scale(cr, 0.45, 1);
        cairo_arc(cr, 0, 0, 13, 0, 2 * M_PI);
        cairo_restore(cr);
        set_rgb(cr, shade(HULL, -0.5, 0.5, 0.3));
        cairo_fill(cr);
        cairo_save(cr);
        cairo_translate(cr, 1.6, 0);
        cairo_scale(cr, 0.34, 1);
        cairo_arc(cr, 0, 0, 12, 0, 2 * M_PI);
        cairo_restore(cr);
        cairo_pattern_t *dg = cairo_pattern_create_linear(0, -12, 0, 12);
        cairo_pattern_add_color_stop_rgb(dg, 0, 0.92, 0.93, 0.95);
        cairo_pattern_add_color_stop_rgb(dg, 1, 0.42, 0.48, 0.58);
        cairo_set_source(cr, dg);
        cairo_fill(cr);
        cairo_pattern_destroy(dg);
        cairo_move_to(cr, 1.6, -9);                 /* feed struts */
        cairo_line_to(cr, 13, 0);
        cairo_line_to(cr, 1.6, 9);
        cairo_set_source_rgba(cr, 0.75, 0.78, 0.82, 0.8);
        cairo_set_line_width(cr, 0.8);
        cairo_stroke(cr);
        cairo_arc(cr, 13, 0, 1.6, 0, 2 * M_PI);
        cairo_set_source_rgb(cr, 0.85, 0.87, 0.9);
        cairo_fill(cr);
        cairo_restore(cr);
    }
    cairo_destroy(cr);
}

/* ---- sprites ---- */

/* The solar wings: a truss through the mast with four panels a side, tilted toward us.
   lit = 0 draws the hardware, lit = 1 only the light of the cells (added on top by power). */
static void solar_point(double X, double Z, double *sx, double *sy)
{
    proj(X, SOLAR_Y - 0.25 * Z, Z, sx, sy);
}

static void draw_solar(cairo_t *cr, int lit)
{
    double x[4], y[4];
    if (!lit) {                                     /* truss */
        double xa, ya, xb, yb;
        solar_point(-SOLAR_OUT - 2, 0, &xa, &ya);
        solar_point(SOLAR_OUT + 2, 0, &xb, &yb);
        cairo_move_to(cr, xa, ya);
        cairo_line_to(cr, xb, yb);
        cairo_set_source_rgb(cr, 0.36, 0.39, 0.45);
        cairo_set_line_width(cr, 2.2);
        cairo_stroke(cr);
    }
    for (int side = -1; side <= 1; side += 2)
        for (int pnl = 0; pnl < 4; pnl++) {
            double seg = (SOLAR_OUT - SOLAR_IN) / 4, X0 = side * (SOLAR_IN + pnl * seg + 1.2), X1 = side * (SOLAR_IN + (pnl + 1) * seg - 1.2);
            solar_point(X0, -SOLAR_W, &x[0], &y[0]);
            solar_point(X1, -SOLAR_W, &x[1], &y[1]);
            solar_point(X1, SOLAR_W, &x[2], &y[2]);
            solar_point(X0, SOLAR_W, &x[3], &y[3]);
            quad(cr, x, y);
            if (!lit) {
                cairo_pattern_t *g = cairo_pattern_create_linear(CX - SOLAR_OUT, y[0], CX + SOLAR_OUT, y[2]);
                cairo_pattern_add_color_stop_rgb(g, 0, 0.16, 0.24, 0.42);    /* sun sheen on the left */
                cairo_pattern_add_color_stop_rgb(g, 0.45, 0.05, 0.09, 0.20);
                cairo_pattern_add_color_stop_rgb(g, 1, 0.04, 0.07, 0.16);
                cairo_set_source(cr, g);
                cairo_fill_preserve(cr);
                cairo_pattern_destroy(g);
                cairo_set_source_rgba(cr, 0.70, 0.74, 0.80, 0.85);
                cairo_set_line_width(cr, 0.7);
                cairo_stroke(cr);
            } else {
                cairo_set_source_rgba(cr, 0.20, 0.50, 1.0, 0.55);
                cairo_fill(cr);
            }
            /* cell grid */
            cairo_new_path(cr);
            for (int i = 1; i < 5; i++) {
                double X = X0 + (X1 - X0) * i / 5, xa, ya, xb, yb;
                solar_point(X, -SOLAR_W, &xa, &ya);
                solar_point(X, SOLAR_W, &xb, &yb);
                cairo_move_to(cr, xa, ya);
                cairo_line_to(cr, xb, yb);
            }
            {
                double xa, ya, xb, yb;
                solar_point(X0, 0, &xa, &ya);
                solar_point(X1, 0, &xb, &yb);
                cairo_move_to(cr, xa, ya);
                cairo_line_to(cr, xb, yb);
            }
            if (lit)
                cairo_set_source_rgba(cr, 0.6, 0.85, 1.0, 0.9);
            else
                cairo_set_source_rgba(cr, 0.55, 0.62, 0.75, 0.35);
            cairo_set_line_width(cr, lit ? 0.8 : 0.5);
            cairo_stroke(cr);
        }
}

static void build_sprites(void)
{
    cairo_t *cr;
    cairo_pattern_t *g;

    solar_lit = new_surf(SIZE, SIZE, &cr);
    draw_solar(cr, 1);
    cairo_destroy(cr);
    pod_glow = make_glow(15, LAMP, 0.55, 0.18);
    for (int k = 0; k < 2; k++) {                   /* anamorphic streak across a hot core */
        streak_spr[k] = new_surf(240, 14, &cr);
        cairo_translate(cr, 120, 7);
        cairo_scale(cr, 1, 7 / 120.0);
        g = cairo_pattern_create_radial(0, 0, 0, 0, 0, 120);
        rgb c = lerp(GPU_COL[k], (rgb){ 1, 1, 1 }, 0.35);
        cairo_pattern_add_color_stop_rgba(g, 0, c.r, c.g, c.b, 0.9);
        cairo_pattern_add_color_stop_rgba(g, 0.25, c.r, c.g, c.b, 0.3);
        cairo_pattern_add_color_stop_rgba(g, 1, c.r, c.g, c.b, 0);
        cairo_set_source(cr, g);
        cairo_paint(cr);
        cairo_pattern_destroy(g);
        cairo_destroy(cr);
    }

    /* reactor cores */
    for (int k = 0; k < 2; k++) {
        rgb c = GPU_COL[k];
        core_dark[k] = new_surf(28, 28, &cr);
        cairo_translate(cr, 14, 14);
        cairo_arc(cr, 0, 0, CORE_R, 0, 2 * M_PI);
        g = cairo_pattern_create_radial(-3, -4, 1, 0, 0, CORE_R);
        cairo_pattern_add_color_stop_rgb(g, 0, 0.55 * c.r + 0.2, 0.55 * c.g + 0.2, 0.55 * c.b + 0.2);
        cairo_pattern_add_color_stop_rgb(g, 0.6, 0.22 * c.r, 0.22 * c.g, 0.22 * c.b + 0.03);
        cairo_pattern_add_color_stop_rgb(g, 1, 0.05, 0.06, 0.09);
        cairo_set_source(cr, g);
        cairo_fill(cr);
        cairo_pattern_destroy(g);
        cairo_destroy(cr);

        core_hot[k] = new_surf(28, 28, &cr);
        cairo_translate(cr, 14, 14);
        cairo_arc(cr, 0, 0, CORE_R, 0, 2 * M_PI);
        g = cairo_pattern_create_radial(0, 0, 0, 0, 0, CORE_R);
        cairo_pattern_add_color_stop_rgb(g, 0, 1, 1, 1);
        cairo_pattern_add_color_stop_rgb(g, 0.35, 0.5 + 0.5 * c.r, 0.5 + 0.5 * c.g, 0.5 + 0.5 * c.b);
        cairo_pattern_add_color_stop_rgb(g, 0.8, c.r, c.g, c.b);
        cairo_pattern_add_color_stop_rgb(g, 1, 0.6 * c.r, 0.6 * c.g, 0.6 * c.b);
        cairo_set_source(cr, g);
        cairo_fill(cr);
        cairo_pattern_destroy(g);
        cairo_destroy(cr);
        react_glow[k] = make_glow(90, c, 0.75, 0.12);
    }

    /* coolant puff */
    puff_spr = new_surf(32, 32, &cr);
    g = cairo_pattern_create_radial(16, 16, 0, 16, 16, 16);
    cairo_pattern_add_color_stop_rgba(g, 0, 0.92, 0.96, 1.0, 0.85);
    cairo_pattern_add_color_stop_rgba(g, 0.4, 0.85, 0.92, 1.0, 0.45);
    cairo_pattern_add_color_stop_rgba(g, 1, 0.8, 0.9, 1.0, 0);
    cairo_set_source(cr, g);
    cairo_paint(cr);
    cairo_pattern_destroy(g);
    cairo_destroy(cr);

    dot_spr[DOT_CYAN]  = make_glow(9, CYAN, 1.0, 0.15);
    dot_spr[DOT_AMBER] = make_glow(9, AMBER, 1.0, 0.15);
    dot_spr[DOT_GREEN] = make_glow(9, (rgb){ 0.4, 1.0, 0.6 }, 1.0, 0.15);
    dot_spr[DOT_RED]   = make_glow(9, (rgb){ 1.0, 0.25, 0.2 }, 1.0, 0.15);

    /* relay satellite: a little bus with two solar wings */
    sat_spr = new_surf(44, 24, &cr);
    cairo_translate(cr, 22, 12);
    cairo_rotate(cr, -0.35);
    for (int s = -1; s <= 1; s += 2) {
        cairo_rectangle(cr, s > 0 ? 4 : -18, -3.5, 14, 7);
        cairo_set_source_rgb(cr, 0.10, 0.16, 0.32);
        cairo_fill_preserve(cr);
        cairo_set_source_rgba(cr, 0.55, 0.65, 0.85, 0.7);
        cairo_set_line_width(cr, 0.6);
        cairo_stroke(cr);
        for (int i = 1; i < 4; i++) {
            double x = (s > 0 ? 4 : -18) + i * 3.5;
            cairo_move_to(cr, x, -3.5);
            cairo_line_to(cr, x, 3.5);
        }
        cairo_set_source_rgba(cr, 0.4, 0.5, 0.8, 0.5);
        cairo_stroke(cr);
    }
    cairo_move_to(cr, -4, 0);
    cairo_line_to(cr, 4, 0);
    cairo_set_line_width(cr, 1);
    cairo_stroke(cr);
    cairo_rectangle(cr, -3.5, -3.5, 7, 7);
    cairo_set_source_rgb(cr, 0.75, 0.66, 0.42);
    cairo_fill(cr);
    cairo_arc(cr, -3, 0, 3, M_PI / 2, 3 * M_PI / 2);
    cairo_set_source_rgb(cr, 0.85, 0.87, 0.9);
    cairo_fill(cr);
    cairo_destroy(cr);

    for (int k = 0; k < N_PAGES; k++)
        hud_cache[k] = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, HUD_H);
}

static void build_world(void)
{
    /* pods fill the deck outside the bay, leaving room at the three spoke junctions */
    static const double arcs[4][2] = { { 30, 55 }, { 65, 175 }, { 185, 295 }, { 305, 330 } };
    static const int counts[4] = { 3, 13, 13, 3 };
    int n = 0;
    for (int a = 0; a < 4; a++)
        for (int j = 0; j < counts[a]; j++)
            pod_ang[n++] = DEG(arcs[a][0] + (j + 0.5) * (arcs[a][1] - arcs[a][0]) / counts[a]);

    srand(11);
    for (int i = 0; i < N_SLOTS; i++)
        slot_ci[i] = rand() % 6;
    for (int i = 0; i < N_SHIPS; i++) {
        ship_order[i] = i * 17 % N_SHIPS;           /* fill the orbit spread out, not in a row */
        ship_r[i] = ORBIT_R + (frand() - 0.5) * 14;
        ship_y[i] = ORBIT_Y + (frand() - 0.5) * 22;
    }
    for (int i = 0; i < N_TWINKLE; i++) {
        double x, y;
        do {
            x = 20 + frand() * 440;
            y = 18 + frand() * 90;
        } while (hypot(x - CX, y - CX) > 228 || hypot(x - 106, y - 62) < 40);
        twinkle[i] = (star){ x, y, 0.6 + frand() * 0.9, frand() * 6.3, 0.5 + frand() * 1.6 };
    }
    srand((unsigned)time(NULL));
}

/* ---------------------------------------------------------------- simulation */

static const double SPOKE_ANG[3] = { 60, 180, 300 };
static const double DOCK_ANG[3]  = { 0, 120, 240 };

/* where a shuttle meets dock k right now, and the outward direction there */
static void dock_tip(int k, double *X, double *Y, double *Z, double *ox, double *oz)
{
    double th = DEG(DOCK_ANG[k]) + rot;
    *ox = cos(th);
    *oz = sin(th);
    *X = (R_OUT + 24) * *ox;
    *Z = (R_OUT + 24) * *oz;
    *Y = -3;
}

static void spawn_shuttle(int dock, int arrive, double activity)
{
    for (int i = 0; i < MAX_SHUTTLES; i++) {
        shuttle *s = &shuttles[i];
        if (s->alive)
            continue;
        s->alive = 1;
        s->dock = dock;
        s->arrive = arrive;
        s->age = 0;
        s->dur = (arrive ? 4.2 : 3.6) - 1.2 * activity + frand() * 0.8;
        s->hold = arrive ? 0.7 : 0.6;
        s->ang = DEG(DOCK_ANG[dock]) + rot + (arrive ? 0.55 : -0.25) + (frand() - 0.5) * 0.5;
        s->y0 = 20 + frand() * 70;
        s->r0 = 380 + frand() * 80;
        s->lx = s->ly = NAN;
        return;
    }
}

/* shuttle position in screen space; returns 0 once it's gone */
static int shuttle_pos(const shuttle *s, double *sx, double *sy, double *alpha, double *thrust)
{
    double X, Y, Z, ox, oz, u, p0[3], p1[3], p2[3];
    dock_tip(s->dock, &X, &Y, &Z, &ox, &oz);
    double travel = s->arrive ? s->age : s->age - s->hold;
    double fade = 1;
    if (s->arrive) {
        if (travel < s->dur)
            u = travel / s->dur, u = 1 - (1 - u) * (1 - u) * (1 - u);          /* brakes on approach */
        else
            u = 1, fade = clamp01(1 - (travel - s->dur - s->hold * 0.5) / (s->hold * 0.5));
        *thrust = travel < s->dur ? 0.4 + 0.6 * (1 - travel / s->dur) : 0;
    } else {
        if (travel < 0)
            u = 0, fade = clamp01(s->age / (s->hold * 0.6));
        else
            u = travel / s->dur, u = u * u;                                     /* accelerates away */
        *thrust = travel < 0 ? 0 : 0.5 + 0.5 * clamp01(travel / s->dur);
    }
    if ((s->arrive ? travel - s->dur - s->hold : travel - s->dur) > 0)
        return 0;
    /* quadratic path: far point, a point straight out from the dock, the dock */
    p0[0] = s->r0 * cos(s->ang);
    p0[1] = s->y0;
    p0[2] = s->r0 * sin(s->ang);
    p1[0] = X + ox * 70;
    p1[1] = Y + 6;
    p1[2] = Z + oz * 70;
    p2[0] = X, p2[1] = Y, p2[2] = Z;
    double v = s->arrive ? u : 1 - u, a = (1 - v) * (1 - v), b = 2 * v * (1 - v), c = v * v;
    double P[3];
    for (int i = 0; i < 3; i++)
        P[i] = a * p0[i] + b * p1[i] + c * p2[i];
    proj(P[0], P[1], P[2], sx, sy);
    double d = hypot(*sx - CX, *sy - CX);
    *alpha = fade * clamp01((236 - d) / 30);
    return 1;
}

static void spawn_puff(double X, double Y, double Z, double ox, double oz, double tang)
{
    for (int i = 0; i < MAX_PUFFS; i++) {
        puff *p = &puffs[i];
        if (p->alive)
            continue;
        double w = 2 * M_PI / ROT_PERIOD;
        p->alive = 1;
        p->x = X, p->y = Y, p->z = Z;
        /* leaves outward and up, keeping the ring's spin speed along the tangent */
        double out = 16 + frand() * 12;
        p->vx = ox * out - oz * w * tang + (frand() - 0.5) * 4;
        p->vz = oz * out + ox * w * tang + (frand() - 0.5) * 4;
        p->vy = 6 + frand() * 8;
        p->age = 0;
        p->life = 2.2 + frand() * 1.4;
        p->size = 0.9 + frand() * 0.5;
        return;
    }
}

static int simulate(const shown_t *sh, double dt)
{
    int active = 0;
    rot += dt * 2 * M_PI / ROT_PERIOD;
    if (rot > 2 * M_PI)
        rot -= 2 * M_PI;

    /* cargo bay: containers for used RAM, ghost crates for page cache */
    double per = sh->mem_total > 1 ? sh->mem_total / N_SLOTS : 91.0 / N_SLOTS;
    double want = sh->mem_used / per;
    if (fabs(want - n_used_slots) > 0.75)           /* hysteresis: no crate flickering in and out */
        n_used_slots = (int)lround(want);
    n_used_slots = n_used_slots < 0 ? 0 : n_used_slots > N_SLOTS ? N_SLOTS : n_used_slots;
    double wantc = sh->mem_cached / per;
    if (fabs(wantc - n_cache_slots) > 0.75)
        n_cache_slots = (int)lround(wantc);
    for (int i = 0; i < N_SLOTS; i++) {
        double target = i < n_used_slots;
        double tg = i >= n_used_slots && i < n_used_slots + n_cache_slots;
        slot_fill[i] += (target - slot_fill[i]) * fmin(1, dt * (target > slot_fill[i] ? 1.8 : 2.5));
        slot_ghost[i] += (tg - slot_ghost[i]) * fmin(1, dt * 2);
    }

    /* shuttles: departures for reads, arrivals for writes, rate ~ log of throughput */
    for (int k = 0; k < SYS_N_NVME; k++) {
        double r = clamp01(log10(1 + sh->rd[k] / 1e6) / log10(1 + 4000));
        double w = clamp01(log10(1 + sh->wr[k] / 1e6) / log10(1 + 4000));
        spawn_rd[k] += (sh->rd[k] > 2e5 ? 0.15 + 2.4 * r : 0) * dt;
        spawn_wr[k] += (sh->wr[k] > 2e5 ? 0.15 + 2.4 * w : 0) * dt;
        if (spawn_rd[k] >= 1) spawn_shuttle(k, 0, r), spawn_rd[k] -= 1;
        if (spawn_wr[k] >= 1) spawn_shuttle(k, 1, w), spawn_wr[k] -= 1;
        if (sh->rd[k] <= 2e5) spawn_rd[k] = fmin(spawn_rd[k], 0.6);
        if (sh->wr[k] <= 2e5) spawn_wr[k] = fmin(spawn_wr[k], 0.6);

        /* hot drives vent from their dock */
        double hv = clamp01((sh->nvme_temp[k] - 44) / 26);
        spawn_vent_nvme[k] += hv * hv * 2.6 * dt;
        while (spawn_vent_nvme[k] >= 1) {
            double X, Y, Z, ox, oz;
            dock_tip(k, &X, &Y, &Z, &ox, &oz);
            spawn_puff(X - ox * 14, Y + 2, Z - oz * 14, ox, oz, R_OUT + 10);
            spawn_vent_nvme[k] -= 1;
        }
    }
    for (int i = 0; i < MAX_SHUTTLES; i++) {
        shuttle *s = &shuttles[i];
        if (!s->alive)
            continue;
        s->age += dt;
        active++;
        double x, y, a, th;
        if (!shuttle_pos(s, &x, &y, &a, &th))
            s->alive = 0;
    }

    /* the CPU vents from its pods as it heats up, busiest pods first */
    double hc = clamp01((sh->cpu_temp - 66) / 28);   /* Tctl idles in the 50s and 60s */
    spawn_vent_cpu += hc * hc * 7 * dt;
    while (spawn_vent_cpu >= 1) {
        int best = rand() % N_PODS;
        for (int tries = 0; tries < 3; tries++) {
            int c = rand() % N_PODS;
            if (sh->pod[c] > sh->pod[best])
                best = c;
        }
        double th = pod_ang[best] + rot;
        spawn_puff(188 * cos(th), 6, 188 * sin(th), cos(th), sin(th), 188);
        spawn_vent_cpu -= 1;
    }
    for (int i = 0; i < MAX_PUFFS; i++) {
        puff *p = &puffs[i];
        if (!p->alive)
            continue;
        p->age += dt;
        p->x += p->vx * dt, p->y += p->vy * dt, p->z += p->vz * dt;
        if (p->age > p->life)
            p->alive = 0;
        active++;
    }

    /* a capsule leaves a pod for every burst of new processes */
    double fr = sh->forks > 0.5 ? fmin(6, 0.9 * log2(1 + sh->forks / 8)) : 0;
    spawn_cap += fr * dt;
    while (spawn_cap >= 1) {
        for (int i = 0; i < MAX_CAPS; i++) {
            capsule *c = &caps[i];
            if (c->alive)
                continue;
            double th = pod_ang[rand() % N_PODS] + rot, sp = 45 + frand() * 30;
            *c = (capsule){ 1, 178 * cos(th), 6, 178 * sin(th), 0, 0, 0, 0, 1.4 + frand() * 0.6, 0 };
            double up = 0.5 + frand() * 0.5;
            c->vx = cos(th) * sp - sin(th) * 12;
            c->vz = sin(th) * sp + cos(th) * 12;
            c->vy = sp * up;
            break;
        }
        spawn_cap -= 1;
    }
    for (int i = 0; i < MAX_CAPS; i++) {
        capsule *c = &caps[i];
        if (!c->alive)
            continue;
        c->age += dt;
        c->x += c->vx * dt, c->y += c->vy * dt, c->z += c->vz * dt;
        if (c->age > c->life)
            c->alive = 0;
        active++;
    }

    /* visiting ships in the holding orbit: one per 3 open TCP connections */
    double wants = fmin(N_SHIPS, sh->tcp / 3);
    if (fabs(wants - n_ships) > 0.75)
        n_ships = (int)lround(wants);
    for (int i = 0; i < N_SHIPS; i++) {
        double tg = 0;
        for (int k = 0; k < n_ships; k++)
            if (ship_order[k] == i)
                tg = 1;
        ship_vis[i] += (tg - ship_vis[i]) * fmin(1, dt * 0.8);
    }
    orbit_rot -= dt * 2 * M_PI / 420;

    /* comms: packets down (rx) and up (tx) the beam */
    double lrx = clamp01(log10(1 + sh->rx / 1e3) / log10(1 + 120000));
    double ltx = clamp01(log10(1 + sh->tx / 1e3) / log10(1 + 120000));
    spawn_rx += (sh->rx > 3000 ? 0.4 + 11 * lrx * lrx : 0) * dt;
    spawn_tx += (sh->tx > 3000 ? 0.4 + 11 * ltx * ltx : 0) * dt;
    for (int k = 0; k < 2; k++) {
        double *acc = k ? &spawn_tx : &spawn_rx;
        while (*acc >= 1) {
            for (int i = 0; i < MAX_PACKETS; i++)
                if (!packets[i].alive) {
                    packets[i] = (packet){ 1, k, 0, 0.55 + frand() * 0.25 };
                    break;
                }
            *acc -= 1;
        }
    }
    for (int i = 0; i < MAX_PACKETS; i++) {
        packet *p = &packets[i];
        if (!p->alive)
            continue;
        p->u += p->speed * dt;
        if (p->u >= 1)
            p->alive = 0;
        active++;
    }

    /* reactor cores: breathe faster with load */
    for (int k = 0; k < N_GPUS; k++) {
        core_phase[k] += dt * (0.6 + 2.6 * sh->load[k]);
        spark_phase[k] += dt * (0.25 + 2.2 * sh->load[k]);
    }
    return active;
}

/* ---------------------------------------------------------------- render */

/* A box on the ring between radii r0..r1, angles a0..a1, heights y0..y1 */
static void draw_box(cairo_t *cr, double r0, double r1, double a0, double a1, double y0, double y1, rgb c, double alpha)
{
    double am = (a0 + a1) / 2, s = sin(am), co = cos(am), x[4], y[4];
    /* radial face: the outer one in front, the inner one at the back */
    double rr = s > 0 ? r1 : r0, ns = s > 0 ? 1 : -1;
    polar(rr, a0, y0, &x[0], &y[0]);
    polar(rr, a1, y0, &x[1], &y[1]);
    polar(rr, a1, y1, &x[2], &y[2]);
    polar(rr, a0, y1, &x[3], &y[3]);
    quad(cr, x, y);
    rgb f = shade(c, ns * co, 0, ns * s);
    cairo_set_source_rgba(cr, f.r, f.g, f.b, alpha);
    cairo_fill(cr);
    /* end face on the side turned toward us */
    double ae = cos(a1) > 0 ? a1 : a0, te = cos(a1) > 0 ? 1 : -1;
    polar(r0, ae, y0, &x[0], &y[0]);
    polar(r1, ae, y0, &x[1], &y[1]);
    polar(r1, ae, y1, &x[2], &y[2]);
    polar(r0, ae, y1, &x[3], &y[3]);
    quad(cr, x, y);
    f = shade(c, -te * sin(ae), 0, te * cos(ae));
    cairo_set_source_rgba(cr, f.r, f.g, f.b, alpha);
    cairo_fill(cr);
    /* top */
    polar(r0, a0, y1, &x[0], &y[0]);
    polar(r1, a0, y1, &x[1], &y[1]);
    polar(r1, a1, y1, &x[2], &y[2]);
    polar(r0, a1, y1, &x[3], &y[3]);
    quad(cr, x, y);
    f = shade(c, 0, 1, 0);
    cairo_set_source_rgba(cr, f.r, f.g, f.b, alpha);
    cairo_fill_preserve(cr);
    cairo_set_source_rgba(cr, 1, 1, 1, 0.12 * alpha);
    cairo_set_line_width(cr, 0.5);
    cairo_stroke(cr);
}

/*
 * Pods and crates look the same wherever they are on the ring except for their angle,
 * so each is pre-rendered at every degree and blitted at its exact position.
 */
#define N_ANG       360
#define POD_AX      15
#define POD_AY      17
#define CRATE_AX    9
#define CRATE_AY    11
static const rgb CRATE[6] = {
    { 0.72, 0.30, 0.20 }, { 0.84, 0.58, 0.22 }, { 0.22, 0.52, 0.55 },
    { 0.30, 0.42, 0.66 }, { 0.82, 0.78, 0.68 }, { 0.55, 0.60, 0.30 },
};
static cairo_surface_t *pod_spr[3][N_ANG], *crate_spr[6][N_ANG];

static int ang_idx(double th)
{
    int i = (int)lround(th * N_ANG / (2 * M_PI)) % N_ANG;
    return i < 0 ? i + N_ANG : i;
}

/* mode 0: the module with dark windows; 1 and 2: only its windows, warm and hot */
static void pod_geom(cairo_t *cr, double th, int mode)
{
    double w = DEG(2.7), x[4], y[4], sx, sy;
    rgb lit = mode == 0 ? (rgb){ 0.16, 0.20, 0.26 } : mode == 1 ? LAMP : LAMP_HOT;
    if (mode == 0)
        draw_box(cr, 169, 187, th - w, th + w, 0, 7, HULL, 1);
    polar(176.5, th - w * 0.8, 7, &x[0], &y[0]);   /* skylight along the roof */
    polar(179.5, th - w * 0.8, 7, &x[1], &y[1]);
    polar(179.5, th + w * 0.8, 7, &x[2], &y[2]);
    polar(176.5, th + w * 0.8, 7, &x[3], &y[3]);
    quad(cr, x, y);
    double rr = sin(th) > 0 ? 187.2 : 168.8;        /* windows on the side facing us */
    for (int i = 0; i < 3; i++) {
        polar(rr, th - w * 0.62 + i * w * 0.62, 3.6, &sx, &sy);
        cairo_rectangle(cr, sx - 0.9, sy - 1.2, 1.8, 2.4);
    }
    set_rgb(cr, lit);
    cairo_fill(cr);
}

static cairo_surface_t *ship_spr[72];

static void build_ring_sprites(void)
{
    cairo_t *cr;
    for (int i = 0; i < 72; i++) {                  /* visiting ships, sunlit from the upper left */
        double th = i * 2 * M_PI / 72, tx = -sin(th) * 4, ty = cos(th) * 4 * TS;
        ship_spr[i] = new_surf(16, 16, &cr);
        cairo_translate(cr, 8, 8);
        cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
        cairo_move_to(cr, tx * 0.9, ty * 0.9);
        cairo_line_to(cr, -tx, -ty);
        cairo_set_line_width(cr, 2.6);
        cairo_set_source_rgb(cr, 0.55, 0.60, 0.68);
        cairo_stroke(cr);
        cairo_move_to(cr, tx * 0.9 - 0.4, ty * 0.9 - 0.6);
        cairo_line_to(cr, -tx - 0.4, -ty - 0.6);
        cairo_set_line_width(cr, 1.0);
        cairo_set_source_rgba(cr, 0.95, 0.95, 0.92, 0.8);
        cairo_stroke(cr);
        cairo_destroy(cr);
    }
    double pitch = 2 * BAY_HALF / BAY_COLS, rc = R_IN + 4.5 + 6.2 + 2.6, px, py;
    for (int a = 0; a < N_ANG; a++) {
        double th = a * 2 * M_PI / N_ANG;
        polar(178, th, 0, &px, &py);
        for (int m = 0; m < 3; m++) {
            pod_spr[m][a] = new_surf(30, 26, &cr);
            cairo_translate(cr, POD_AX - px, POD_AY - py);
            pod_geom(cr, th, m);
            cairo_destroy(cr);
        }
        polar(rc, th, 0, &px, &py);
        for (int c = 0; c < 6; c++) {
            crate_spr[c][a] = new_surf(18, 16, &cr);
            cairo_translate(cr, CRATE_AX - px, CRATE_AY - py);
            draw_box(cr, rc - 2.6, rc + 2.6, th - pitch * 0.38, th + pitch * 0.38, 0, 5.5, CRATE[c], 1);
            cairo_destroy(cr);
        }
    }
}

/* blit sprite s with its anchor (ax, ay) on (x, y) */
static void spr_at(cairo_t *cr, cairo_surface_t *s, double x, double y, double ax, double ay, double alpha)
{
    if (alpha <= 0.004)
        return;
    int w = cairo_image_surface_get_width(s), h = cairo_image_surface_get_height(s);
    cairo_set_source_surface(cr, s, x - ax, y - ay);
    cairo_rectangle(cr, floor(x - ax), floor(y - ay), w + 1, h + 1);
    if (alpha >= 0.996) {
        cairo_fill(cr);
    } else {
        cairo_save(cr);
        cairo_clip(cr);
        cairo_paint_with_alpha(cr, alpha);
        cairo_restore(cr);
    }
}

enum { IT_POD, IT_CRATE, IT_JUNCTION };
typedef struct { int type, idx; double depth, ang; } item;

static int by_depth(const void *a, const void *b)
{
    double da = ((const item *)a)->depth, db = ((const item *)b)->depth;
    return da < db ? -1 : da > db;
}

static void draw_item(cairo_t *cr, const item *it, const shown_t *sh, double t)
{
    double th = it->ang, sx, sy;
    if (it->type == IT_POD) {
        /* a habitat module: idle ones keep a dim nightlight, busy ones glow warm, then white-hot */
        double l = sh->pod[it->idx];
        int ai = ang_idx(th);
        polar(178, th, 0, &sx, &sy);
        spr_at(cr, pod_spr[0][ai], sx, sy, POD_AX, POD_AY, 1);
        spr_at(cr, pod_spr[1][ai], sx, sy, POD_AX, POD_AY, clamp01(l * 1.7));
        spr_at(cr, pod_spr[2][ai], sx, sy, POD_AX, POD_AY, clamp01((l - 0.55) / 0.45));
    } else if (it->type == IT_CRATE) {
        int i = it->idx, col = i / BAY_LANES, lane = i % BAY_LANES;
        double f = slot_fill[i], gh = slot_ghost[i];
        double pitch = 2 * BAY_HALF / BAY_COLS, tc = th + col * pitch - BAY_HALF + pitch * 0.5;
        double r0 = R_IN + 4.5 + lane * 6.2, r1 = r0 + 5.2;
        if (f > 0.02) {
            double drop = (1 - f) * (1 - f) * 22;   /* crates are lowered in from above */
            polar(r0 + 2.6, tc, drop, &sx, &sy);
            spr_at(cr, crate_spr[slot_ci[i]][ang_idx(tc)], sx, sy, CRATE_AX, CRATE_AY, clamp01(f * 1.4));
        } else if (gh > 0.02) {
            double x[4], y[4], a0 = tc - pitch * 0.38, a1 = tc + pitch * 0.38;
            polar(r0, a0, 0.4, &x[0], &y[0]);
            polar(r1, a0, 0.4, &x[1], &y[1]);
            polar(r1, a1, 0.4, &x[2], &y[2]);
            polar(r0, a1, 0.4, &x[3], &y[3]);
            quad(cr, x, y);
            cairo_set_source_rgba(cr, 0.55, 0.8, 1.0, 0.35 * gh);
            cairo_set_line_width(cr, 0.7);
            cairo_stroke(cr);
        }
    } else {
        double w = 5.0 / 178;
        draw_box(cr, R_IN + 1, R_OUT - 1, th - w, th + w, 0, 5, HULL, 1);
        /* CPU pressure: red warning strobes on the junctions */
        double pc = clamp01((sh->psi_cpu - 2) / 28);
        if (pc > 0.01) {
            polar(178, th, 6, &sx, &sy);
            double st = pow(0.5 + 0.5 * sin(t * (3 + 4 * pc) + it->idx * 2), 4);
            blit_add(cr, dot_spr[DOT_RED], sx, sy, 0.9 + 0.6 * pc, pc * (0.35 + 0.65 * st));
        }
    }
}

static void draw_dock(cairo_t *cr, int k, const shown_t *sh, double t)
{
    double th = DEG(DOCK_ANG[k]) + rot, w = 3.2 / R_OUT, sx, sy;
    draw_box(cr, R_OUT - 2, R_OUT + 16, th - w, th + w, -6, 0, HULL_DK, 1);           /* arm */
    double wc = 7.5 / (R_OUT + 18);
    draw_box(cr, R_OUT + 16, R_OUT + 21, th - wc, th + wc, -8, 2, HULL, 1);           /* collar */
    /* activity lights on the arm: cyan for reads, amber for writes */
    polar(R_OUT + 8, th, 0.5, &sx, &sy);
    double ar = clamp01(log10(1 + sh->rd[k] / 1e6) / 3), aw = clamp01(log10(1 + sh->wr[k] / 1e6) / 3);
    blit_add(cr, dot_spr[DOT_CYAN], sx, sy, 0.8, 0.15 + 0.8 * ar);
    polar(R_OUT + 13, th, 0.5, &sx, &sy);
    blit_add(cr, dot_spr[DOT_AMBER], sx, sy, 0.8, 0.15 + 0.8 * aw);
    /* docking beacon: a slow green blink, offset per dock */
    polar(R_OUT + 19, th, 2.5, &sx, &sy);
    double b = pow(0.5 + 0.5 * sin(t * 2.1 + k * 2.1), 6);
    double pio = clamp01((sh->psi_io - 2) / 28);        /* I/O pressure turns the beacon red */
    double rb = pow(0.5 + 0.5 * sin(t * 6 + k * 2.1), 4);
    blit_add(cr, dot_spr[DOT_GREEN], sx, sy, 0.9, (1 - pio) * (0.15 + 0.85 * b));
    blit_add(cr, dot_spr[DOT_RED], sx, sy, 0.9 + 0.5 * pio, pio * (0.3 + 0.7 * rb));
}

static void draw_spoke(cairo_t *cr, double th)
{
    double w = 3.4, x[4], y[4];
    double ux = cos(th), uz = sin(th), nx = -uz, nz = ux;       /* along and across */
    double side = nz > 0 ? 1 : -1;                              /* the side facing us */
    double r0 = HUB_R - 2, r1 = R_IN + 1;
    for (int face = 0; face < 2; face++) {
        double o = face == 0 ? side * w : -w, o2 = face == 0 ? side * w : w;
        double ya0 = face == 0 ? -7 : -1.5, ya1 = face == 0 ? -1.5 : -1.5;
        proj(ux * r0 + nx * o, ya0, uz * r0 + nz * o, &x[0], &y[0]);
        proj(ux * r1 + nx * o, ya0, uz * r1 + nz * o, &x[1], &y[1]);
        proj(ux * r1 + nx * o2, ya1, uz * r1 + nz * o2, &x[2], &y[2]);
        proj(ux * r0 + nx * o2, ya1, uz * r0 + nz * o2, &x[3], &y[3]);
        quad(cr, x, y);
        rgb f = face == 0 ? shade(HULL_DK, side * nx, 0, side * nz) : shade(HULL, 0, 1, 0);
        set_rgb(cr, f);
        cairo_fill(cr);
    }
    /* a lit conduit along the top */
    double xa, ya, xb, yb;
    proj(ux * (r0 + 4), -1.2, uz * (r0 + 4), &xa, &ya);
    proj(ux * (r1 - 3), -1.2, uz * (r1 - 3), &xb, &yb);
    cairo_move_to(cr, xa, ya);
    cairo_line_to(cr, xb, yb);
    cairo_set_source_rgba(cr, 1, 0.97, 0.9, 0.22);
    cairo_set_line_width(cr, 0.8);
    cairo_stroke(cr);
}

static void draw_reactor(cairo_t *cr, int k, const shown_t *sh, int front)
{
    double Yr = k ? REACT_Y1 : REACT_Y0, cx, cy;
    proj(0, Yr, 0, &cx, &cy);
    double p = clamp01((sh->power[k] - GPU_IDLE_W * 0.6) / (GPU_MAX_W - GPU_IDLE_W * 0.6));
    double pulse = 0.5 + 0.5 * sin(core_phase[k]);
    rgb c = GPU_COL[k];
    if (!front) {
        /* radiator panels, unfolding with the GPU's fan speed; stripes glow with its temperature */
        double f = smooth01(0.08 + sh->fan[k] / 100.0), fold = (k ? 1 : -1) * DEG(82) * (1 - f);
        double heat = clamp01((sh->temp[k] - 38) / 45);
        for (int side = -1; side <= 1; side += 2) {
            double hx = side * (CONT_R + 1), dxw = side * cos(fold), dyw = sin(fold), x[4], y[4];
            proj(hx, Yr, -7.0, &x[0], &y[0]);
            proj(hx + dxw * RAD_LEN, Yr + dyw * RAD_LEN, -7.0, &x[1], &y[1]);
            proj(hx + dxw * RAD_LEN, Yr + dyw * RAD_LEN, 7.0, &x[2], &y[2]);
            proj(hx, Yr, 7.0, &x[3], &y[3]);
            quad(cr, x, y);
            rgb pc = shade((rgb){ 0.80, 0.82, 0.86 }, 0, cos(fold) * 0.9, 0.45);
            set_rgb(cr, pc);
            cairo_fill_preserve(cr);
            cairo_set_source_rgba(cr, 0.1, 0.12, 0.16, 0.6);
            cairo_set_line_width(cr, 0.6);
            cairo_stroke(cr);
            rgb hc = heat_color(0.45 + 0.55 * heat);
            for (int i = 0; i < 3; i++) {
                double zz = -4.5 + i * 4.5, xa, ya, xb, yb;
                proj(hx + dxw * 2, Yr + dyw * 2, zz, &xa, &ya);
                proj(hx + dxw * (RAD_LEN - 2), Yr + dyw * (RAD_LEN - 2), zz, &xb, &yb);
                cairo_move_to(cr, xa, ya);
                cairo_line_to(cr, xb, yb);
            }
            cairo_set_source_rgba(cr, hc.r, hc.g, hc.b, 0.25 + 0.75 * heat);
            cairo_set_line_width(cr, 0.9);
            cairo_stroke(cr);
        }
        blit_add(cr, react_glow[k], cx, cy, 0.42 + 0.58 * p, 0.10 + 0.55 * p + 0.10 * pulse * p);
        blit(cr, core_dark[k], cx, cy, 1, 1);
        blit(cr, core_hot[k], cx, cy, 1, clamp01(0.12 + 0.88 * p * (0.82 + 0.18 * pulse)));
        /* sparks orbiting behind the core */
        for (int i = 0; i < 3; i++) {
            double a = spark_phase[k] * (k ? -1 : 1) + i * 2 * M_PI / 3, sx, sy;
            if (sin(a) >= 0)
                continue;
            polar(CONT_R, a, Yr, &sx, &sy);
            blit_add(cr, dot_spr[k ? DOT_AMBER : DOT_CYAN], sx, sy, 0.7, 0.25 + 0.7 * p);
        }
        return;
    }
    ring_arc(cr, CONT_R, Yr, 0, M_PI, 1);
    cairo_set_line_width(cr, 3.2);
    rgb rc = lerp(shade(HULL, 0, 0.3, 1), c, 0.25 * p);
    set_rgb(cr, rc);
    cairo_stroke(cr);
    ring_arc(cr, CONT_R, Yr + 1.2, DEG(20), DEG(160), 1);
    cairo_set_source_rgba(cr, c.r, c.g, c.b, 0.25 + 0.6 * p);
    cairo_set_line_width(cr, 0.9);
    cairo_stroke(cr);
    for (int i = 0; i < 3; i++) {
        double a = spark_phase[k] * (k ? -1 : 1) + i * 2 * M_PI / 3, sx, sy;
        if (sin(a) < 0)
            continue;
        polar(CONT_R, a, Yr, &sx, &sy);
        blit_add(cr, dot_spr[k ? DOT_AMBER : DOT_CYAN], sx, sy, 0.8, 0.3 + 0.7 * p);
    }
    if (p > 0.3)                                    /* a lens streak when it runs hot */
        blit_add(cr, streak_spr[k], cx, cy, 1, smooth01((p - 0.3) / 0.7) * (0.5 + 0.12 * pulse));
}

/* ---- HUD ---- */

typedef struct { const char *s; double size; rgb c; int bold; } seg;

static void soft_text_at(cairo_t *cr, double x, double base, const seg *g)
{
    cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, g->bold ? CAIRO_FONT_WEIGHT_BOLD : CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, g->size);
    cairo_move_to(cr, x, base);
    cairo_text_path(cr, g->s);
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
    cairo_set_line_width(cr, g->size * 0.2);
    cairo_set_source_rgba(cr, 0.0, 0.01, 0.05, 0.8);
    cairo_stroke_preserve(cr);
    set_rgb(cr, g->c);
    cairo_fill(cr);
}

/* A line of segments centred on the screen, shrunk if it would leave the circle */
static void text_line(cairo_t *cr, double base, seg *g, int n)
{
    double w = 0, top = base, adv[8];
    cairo_text_extents_t e;
    for (int pass = 0; pass < 2; pass++) {
        w = 0;
        top = base;
        for (int i = 0; i < n; i++) {
            cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, g[i].bold ? CAIRO_FONT_WEIGHT_BOLD : CAIRO_FONT_WEIGHT_NORMAL);
            cairo_set_font_size(cr, g[i].size);
            cairo_text_extents(cr, g[i].s, &e);
            adv[i] = e.x_advance;
            w += adv[i];
            top = fmin(top, base - g[i].size * 0.75);
        }
        /* the chord at the lower edge of the text (below the centre it is the narrower) */
        double dy = fmax(fabs(base + 4 - CX), fabs(top - CX));
        double chord = 2 * sqrt(fmax(0, 232.0 * 232.0 - dy * dy)) - 24;
        if (w <= chord || pass)
            break;
        double k = chord / w;
        for (int i = 0; i < n; i++)
            g[i].size = fmax(22, g[i].size * k);
    }
    double x = CX - w / 2;
    for (int i = 0; i < n; i++) {
        soft_text_at(cr, x, base, &g[i]);
        x += adv[i];
    }
}

/* bytes/s as a short string */
static void fmt_rate(char *b, size_t n, double v)
{
    if (v >= 1e9)
        snprintf(b, n, "%.1f GB/s", v / 1e9);
    else if (v >= 99.5e6)
        snprintf(b, n, "%.0f MB/s", v / 1e6);
    else if (v >= 1e6)
        snprintf(b, n, "%.1f MB/s", v / 1e6);
    else if (v >= 1e3)
        snprintf(b, n, "%.0f KB/s", v / 1e3);
    else
        snprintf(b, n, "0 KB/s");
}

/* move a shown number only when the value has left a band around it */
static double hyst(double *shown, double v, double band)
{
    if (isnan(*shown) || fabs(v - *shown) > band)
        *shown = v;
    return *shown;
}

/* a rate held with relative hysteresis (4%), so its digits settle */
static double hyst_rate(double *shown, double v)
{
    if (isnan(*shown) || fabs(v - *shown) > fmax(2e3, 0.04 * fmax(v, *shown)))
        *shown = v;
    return *shown;
}

static void update_hud(int page, const shown_t *sh, double t)
{
    static double next, v_cpu = NAN, v_temp = NAN, v_mem = NAN, v_w = NAN, v_t0 = NAN, v_t1 = NAN,
                  v_tok = NAN, v_net = NAN, v_disk = NAN, v_cw = NAN, v_tcp = NAN, v_fork = NAN;
    char key[160], a[48], b[48], c[48];
    if (t >= next || t < next - 1) {
        next = t + 0.25;
        double disk = 0;
        for (int k = 0; k < SYS_N_NVME; k++)
            disk += sh->rd[k] + sh->wr[k];
        hyst(&v_cpu, sh->cpu * 100, 0.8);
        hyst(&v_temp, sh->cpu_temp, 0.8);
        hyst(&v_mem, sh->mem_used, 0.6);
        hyst(&v_w, sh->power[0] + sh->power[1], 4);
        hyst(&v_t0, sh->temp[0], 0.8);
        hyst(&v_t1, sh->temp[1], 0.8);
        hyst(&v_tok, sh->tok, 3);
        hyst_rate(&v_net, sh->rx + sh->tx);
        hyst_rate(&v_disk, disk);
        hyst(&v_cw, sh->cpu_watts, 2);
        hyst(&v_tcp, sh->tcp, 0.9);
        hyst(&v_fork, sh->forks, fmax(1.5, 0.06 * sh->forks));
    }
    const rgb white = { 0.97, 0.97, 1.0 }, dim = { 0.62, 0.72, 0.86 }, soft = { 0.86, 0.89, 0.95 };
    rgb blue = lerp(GPU_COL[0], white, 0.25), orange = lerp(GPU_COL[1], white, 0.15);
    seg l1[4], l2[5];
    int n1 = 0, n2 = 0;
    const char *mid = "   ";

    if (page == 0) {
        snprintf(a, sizeof(a), "%.0f%%", v_cpu);
        snprintf(b, sizeof(b), isnan(v_temp) ? "--\xC2\xB0" "C" : "%.0f\xC2\xB0" "C", v_temp);
        snprintf(c, sizeof(c), "%.0f W", v_cw);
        l1[n1++] = (seg){ "CPU ", 24, dim, 1 };
        l1[n1++] = (seg){ a, 40, white, 1 };
        l2[n2++] = (seg){ b, 24, heat_color((v_temp - 40) / 50), 1 };
        if (sh->watts_real) {                       /* package watts only when RAPL is readable */
            l2[n2++] = (seg){ mid, 24, dim, 0 };
            l2[n2++] = (seg){ c, 24, soft, 1 };
        }
    } else if (page == 1) {
        snprintf(a, sizeof(a), "%.0f W", v_w);
        snprintf(b, sizeof(b), "%.0f\xC2\xB0", v_t0);
        snprintf(c, sizeof(c), "%.0f\xC2\xB0", v_t1);
        l1[n1++] = (seg){ "GPU ", 24, dim, 1 };
        l1[n1++] = (seg){ a, 40, white, 1 };
        l2[n2++] = (seg){ b, 24, blue, 1 };
        static char tk[48];
        if (v_tok >= 1) {
            snprintf(tk, sizeof(tk), "   %.0f%s   ", shown_rate(v_tok), rate_suffix());
            l2[n2++] = (seg){ tk, 22, soft, 0 };
        } else {
            l2[n2++] = (seg){ "      ", 24, dim, 0 };
        }
        l2[n2++] = (seg){ c, 24, orange, 1 };
    } else if (page == 3) {
        snprintf(a, sizeof(a), "%.0f GB", v_mem);
        snprintf(b, sizeof(b), "%.0f TCP", v_tcp);
        snprintf(c, sizeof(c), "%.0f forks/s", v_fork);
        l1[n1++] = (seg){ "RAM ", 24, dim, 1 };
        l1[n1++] = (seg){ a, 40, white, 1 };
        l2[n2++] = (seg){ b, 22, soft, 1 };
        l2[n2++] = (seg){ mid, 22, dim, 0 };
        l2[n2++] = (seg){ c, 22, soft, 1 };
    } else {
        fmt_rate(a, sizeof(a), v_net);
        fmt_rate(b, sizeof(b), v_disk);
        l1[n1++] = (seg){ "NET ", 24, dim, 1 };
        l1[n1++] = (seg){ a, 36, white, 1 };
        l2[n2++] = (seg){ "DISK ", 22, dim, 1 };
        l2[n2++] = (seg){ b, 24, soft, 1 };
    }
    snprintf(key, sizeof(key), "%d", page);
    for (int i = 0; i < n1; i++)
        snprintf(key + strlen(key), sizeof(key) - strlen(key), "|%s", l1[i].s);
    for (int i = 0; i < n2; i++)
        snprintf(key + strlen(key), sizeof(key) - strlen(key), "|%s", l2[i].s);
    if (page == 0)
        snprintf(key + strlen(key), sizeof(key) - strlen(key), "|%.0f", v_temp);
    if (!strcmp(key, hud_key[page]))
        return;
    strcpy(hud_key[page], key);

    cairo_t *cr = cairo_create(hud_cache[page]);
    cairo_set_operator(cr, CAIRO_OPERATOR_CLEAR);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    cairo_translate(cr, 0, -HUD_Y);
    text_line(cr, 408, l1, n1);
    text_line(cr, 440, l2, n2);
    cairo_destroy(cr);
}

/* visiting ships in the holding orbit, the half behind the station or the half in front */
static void draw_ships(cairo_t *cr, int front, double t)
{
    for (int i = 0; i < N_SHIPS; i++) {
        double a = ship_vis[i], th = orbit_rot + i * 2 * M_PI / N_SHIPS + 0.05 * sin(i * 7.1), sx, sy;
        if (a < 0.02 || (sin(th) >= 0) != front)
            continue;
        polar(ship_r[i], th, ship_y[i], &sx, &sy);
        a *= clamp01((234 - hypot(sx - CX, sy - CX)) / 16);
        if (a < 0.02)
            continue;
        /* a small hull pointing along the orbit (pre-rendered per 5 degrees) */
        int si = (int)lround(th / (2 * M_PI) * 72) % 72;
        si = si < 0 ? si + 72 : si;
        spr_at(cr, ship_spr[si], sx, sy, 8, 8, a);
        double ex = sx + 4 * sin(th), ey = sy - 4 * cos(th) * TS;
        double bl = pow(0.5 + 0.5 * sin(t * 1.9 + i * 1.3), 12);
        blit_add(cr, dot_spr[i & 1 ? DOT_RED : DOT_GREEN], ex, ey, 0.5, a * (0.15 + 0.85 * bl));
    }
}

/* Mission clock from uptime at the top, and a pressure warning under it when something stalls */
static cairo_surface_t *top_cache;
static void update_top(const shown_t *sh)
{
    static char key[96];
    static int warn;
    static double shown_psi = NAN;
    char k[96], day[48], w[48] = "";
    double psi[3] = { sh->psi_cpu, sh->psi_io, sh->psi_mem };
    static const char *what[3] = { "CPU", "I/O", "MEMORY" };
    int worst = 0;
    for (int i = 1; i < 3; i++)
        if (psi[i] > psi[worst])
            worst = i;
    if (psi[worst] >= 10)                           /* on at 10%, off below 7% */
        warn = 1;
    else if (psi[worst] < 7)
        warn = 0;
    long up = (long)sh->uptime;
    snprintf(day, sizeof(day), "DAY %ld  \xC2\xB7  %02ld:%02ld", up / 86400 + 1, up / 3600 % 24, up / 60 % 60);
    if (warn)
        snprintf(w, sizeof(w), "%s PRESSURE %.0f%%", what[worst], hyst(&shown_psi, psi[worst], 1.2));
    snprintf(k, sizeof(k), "%s|%s", day, w);
    if (!strcmp(k, key))
        return;
    strcpy(key, k);
    if (!top_cache)
        top_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, 84);
    cairo_t *cr = cairo_create(top_cache);
    cairo_set_operator(cr, CAIRO_OPERATOR_CLEAR);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    seg a = { day, 22, { 0.70, 0.80, 0.94 }, 1 };
    text_line(cr, 42, &a, 1);
    if (*w) {
        seg b = { w, 22, { 1.0, 0.36, 0.28 }, 1 };
        text_line(cr, 70, &b, 1);
    }
    cairo_destroy(cr);
}

static void render(cairo_t *cr, const shown_t *sh, double t, int page, int nx, double fade)
{
    static item items[N_PODS + N_SLOTS + 3];
    int n = 0;

    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    cairo_set_source_surface(cr, bg_layer, 0, 0);
    cairo_paint(cr);

    /* twinkling stars */
    for (int i = 0; i < N_TWINKLE; i++) {
        const star *st = &twinkle[i];
        double a = 0.5 + 0.5 * sin(t * st->sp + st->ph);
        cairo_arc(cr, st->x, st->y, st->r, 0, 2 * M_PI);
        cairo_set_source_rgba(cr, 0.85, 0.9, 1, a * a * 0.8);
        cairo_fill(cr);
    }

    /* relay satellite and the comms beam */
    double net = clamp01(log10(1 + (sh->rx + sh->tx) / 1e3) / log10(1 + 120000));
    double dx, dy;
    proj(0, DISH_Y, 0, &dx, &dy);
    dy -= 2;
    double bx0 = dx + 11 * cos(atan2(SAT_Y - dy, SAT_X - dx)), by0 = dy + 11 * sin(atan2(SAT_Y - dy, SAT_X - dx));
    blit(cr, sat_spr, SAT_X, SAT_Y, 1, 1);
    blit_add(cr, dot_spr[DOT_RED], SAT_X - 1, SAT_Y + 1, 0.8, 0.2 + 0.8 * pow(0.5 + 0.5 * sin(t * 1.7), 8));

    draw_ships(cr, 0, t);

    /* rotating details on the deck and the back inner wall */
    cairo_new_path(cr);
    for (int i = 0; i < 72; i++) {
        double th = rot + i * 2 * M_PI / 72, x0, y0, x1, y1;
        polar(R_IN + 3.5, th, 0, &x0, &y0);
        polar(R_OUT - 3.5, th, 0, &x1, &y1);
        cairo_move_to(cr, x0, y0);
        cairo_line_to(cr, x1, y1);
    }
    cairo_set_source_rgba(cr, 0.02, 0.03, 0.07, 0.30);
    cairo_set_line_width(cr, 0.8);
    cairo_stroke(cr);
    double lights = 0.35 + 0.65 * sh->cpu;
    for (int i = 0; i < 90; i++) {
        double th = rot + i * 2 * M_PI / 90 + 0.02, x, y;
        if (sin(th) > -0.12 || ((i * 7919) % 13) < 3)
            continue;
        polar(R_IN, th, -RING_H * 0.45, &x, &y);
        cairo_rectangle(cr, x - 1.0, y - 1.3, 2.0, 2.6);
    }
    cairo_set_source_rgba(cr, LAMP.r, LAMP.g, LAMP.b, 0.35 + 0.55 * lights);
    cairo_fill(cr);

    /* cargo bay floor */
    {
        double a0 = rot - BAY_HALF - 0.02, a1 = rot + BAY_HALF + 0.02;
        cairo_new_path(cr);
        ring_arc(cr, R_IN + 2.5, 0, a0, a1, 1);
        ring_arc(cr, R_OUT - 2.5, 0, a1, a0, -1);
        cairo_close_path(cr);
        cairo_set_source_rgba(cr, 0.05, 0.06, 0.09, 0.85);
        cairo_fill(cr);
        double pm = clamp01((sh->psi_mem - 3) / 27) * (0.6 + 0.4 * sin(t * 4));
        rgb el = lerp(LAMP, (rgb){ 1.0, 0.18, 0.12 }, pm);    /* memory pressure: red bay lights */
        for (int k = 0; k < 2; k++) {
            ring_arc(cr, k ? R_OUT - 3 : R_IN + 3, 0, a0, a1, 1);
            cairo_set_source_rgba(cr, el.r, el.g, el.b, 0.55 + 0.4 * pm);
            cairo_set_line_width(cr, 0.9);
            cairo_stroke(cr);
        }
    }

    /* everything on the ring, sorted by depth */
    for (int i = 0; i < N_PODS; i++) {
        double th = pod_ang[i] + rot;
        items[n++] = (item){ IT_POD, i, 178 * sin(th) * TC, th };
    }
    for (int i = 0; i < N_SLOTS; i++) {
        int col = i / BAY_LANES, lane = i % BAY_LANES;
        double pitch = 2 * BAY_HALF / BAY_COLS, th = rot + col * pitch - BAY_HALF + pitch * 0.5;
        double r = R_IN + 7 + lane * 6.2;
        if (slot_fill[i] > 0.02 || slot_ghost[i] > 0.02)
            items[n++] = (item){ IT_CRATE, i, r * sin(th) * TC, rot };
    }
    for (int i = 0; i < 3; i++) {
        double th = DEG(SPOKE_ANG[i]) + rot;
        items[n++] = (item){ IT_JUNCTION, i, 178 * sin(th) * TC, th };
    }
    qsort(items, n, sizeof(item), by_depth);
    int j = 0;
    for (; j < n && items[j].depth < 0; j++)
        draw_item(cr, &items[j], sh, t);
    for (int k = 0; k < 3; k++)
        if (sin(DEG(DOCK_ANG[k]) + rot) < 0)
            draw_dock(cr, k, sh, t);

    /* back spokes, lower mast and core, hub, front spokes, upper mast and core */
    for (int k = 0; k < 3; k++)
        if (sin(DEG(SPOKE_ANG[k]) + rot) < 0)
            draw_spoke(cr, DEG(SPOKE_ANG[k]) + rot);
    paint_layer(cr, mast_lo_layer, CX - 26, 250, 52, 100);
    draw_reactor(cr, 1, sh, 0);
    draw_reactor(cr, 1, sh, 1);
    paint_layer(cr, hub_layer, CX - HUB_R - 2, 180, HUB_R * 2 + 4, 80);
    for (int k = 0; k < 3; k++)
        if (sin(DEG(SPOKE_ANG[k]) + rot) >= 0)
            draw_spoke(cr, DEG(SPOKE_ANG[k]) + rot);
    paint_layer(cr, mast_hi_layer, CX - SOLAR_OUT - 4, 118, 2 * SOLAR_OUT + 8, 100);
    {   /* the solar wings light up with CPU package power; energy runs in along the truss */
        double wv = clamp01(sh->cpu_watts / 200);
        cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
        cairo_set_source_surface(cr, solar_lit, 0, 0);
        cairo_rectangle(cr, CX - SOLAR_OUT - 4, 130, 2 * SOLAR_OUT + 8, 36);
        cairo_clip(cr);
        cairo_paint_with_alpha(cr, 0.06 + 0.5 * wv);
        cairo_reset_clip(cr);
        cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
        int np = 2 + (int)(wv * 6);
        for (int i = 0; i < np; i++)
            for (int side = -1; side <= 1; side += 2) {
                double u = fmod(t * (0.25 + 0.35 * wv) + i / (double)np + (side > 0 ? 0.5 / np : 0), 1.0), sx, sy;
                solar_point(side * (SOLAR_OUT - u * (SOLAR_OUT - 6)), 0, &sx, &sy);
                blit_add(cr, dot_spr[DOT_CYAN], sx, sy, 0.55, (0.2 + 0.8 * wv) * sin(u * M_PI));
            }
    }
    draw_reactor(cr, 0, sh, 0);
    draw_reactor(cr, 0, sh, 1);

    /* comms beam and packets, from the dish to the satellite */
    {
        cairo_move_to(cr, bx0, by0);
        cairo_line_to(cr, SAT_X - 4, SAT_Y + 2);
        cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
        cairo_set_source_rgba(cr, 0.35, 0.7, 1.0, 0.05 + 0.16 * net);
        cairo_set_line_width(cr, 3.5);
        cairo_stroke_preserve(cr);
        cairo_set_source_rgba(cr, 0.6, 0.85, 1.0, 0.08 + 0.25 * net);
        cairo_set_line_width(cr, 0.8);
        cairo_stroke(cr);
        cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
        for (int i = 0; i < MAX_PACKETS; i++) {
            const packet *p = &packets[i];
            if (!p->alive)
                continue;
            double u = p->up ? p->u : 1 - p->u;         /* rx comes down from the satellite */
            double x = bx0 + (SAT_X - 4 - bx0) * u, y = by0 + (SAT_Y + 2 - by0) * u;
            double a = clamp01(p->u * 8) * clamp01((1 - p->u) * 8);
            blit_add(cr, dot_spr[p->up ? DOT_AMBER : DOT_CYAN], x, y, 0.75, a);
        }
        blit_add(cr, dot_spr[DOT_CYAN], bx0, by0, 0.7, 0.15 + 0.7 * net);
    }

    /* front: outer wall, its windows, docks, then things on the front of the deck */
    paint_layer(cr, wall_layer, CX - R_OUT - 2, CY - 4, 2 * R_OUT + 4, R_OUT * TS + RING_H * TC + 8);
    for (int i = 0; i < 96; i++) {
        double th = rot + i * 2 * M_PI / 96 + 0.03, x, y;
        if (sin(th) < 0.12 || ((i * 104729) % 11) < 3)
            continue;
        polar(R_OUT, th, -RING_H * 0.30, &x, &y);
        cairo_rectangle(cr, x - 1.1, y - 1.4, 2.2, 2.8);
    }
    cairo_set_source_rgba(cr, LAMP.r, LAMP.g, LAMP.b, 0.3 + 0.6 * lights);
    cairo_fill(cr);
    for (int k = 0; k < 3; k++)
        if (sin(DEG(DOCK_ANG[k]) + rot) >= 0)
            draw_dock(cr, k, sh, t);
    for (; j < n; j++)
        draw_item(cr, &items[j], sh, t);

    /* pod glows, all at once and additive */
    cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
    for (int i = 0; i < N_PODS; i++) {
        double l = sh->pod[i], sx, sy;
        if (l < 0.04)
            continue;
        polar(178, pod_ang[i] + rot, 0, &sx, &sy);
        blit(cr, pod_glow, sx, sy - 5, 0.7 + 0.5 * l, 0.15 + 0.85 * l * l);
    }
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);

    draw_ships(cr, 1, t);

    /* escape capsules: one per burst of new processes */
    for (int i = 0; i < MAX_CAPS; i++) {
        const capsule *c = &caps[i];
        if (!c->alive)
            continue;
        double sx, sy, ex, ey, u = c->age / c->life;
        proj(c->x, c->y, c->z, &sx, &sy);
        proj(c->x - c->vx * 0.22, c->y - c->vy * 0.22, c->z - c->vz * 0.22, &ex, &ey);
        double a = clamp01(u * 10) * (1 - u) * clamp01((236 - hypot(sx - CX, sy - CX)) / 20);
        cairo_pattern_t *g = cairo_pattern_create_linear(sx, sy, ex, ey);
        cairo_pattern_add_color_stop_rgba(g, 0, 0.75, 1.0, 0.85, 0.8 * a);
        cairo_pattern_add_color_stop_rgba(g, 1, 0.75, 1.0, 0.85, 0);
        cairo_move_to(cr, sx, sy);
        cairo_line_to(cr, ex, ey);
        cairo_set_source(cr, g);
        cairo_set_line_width(cr, 1.3);
        cairo_stroke(cr);
        cairo_pattern_destroy(g);
        cairo_arc(cr, sx, sy, 1.3, 0, 2 * M_PI);
        cairo_set_source_rgba(cr, 0.95, 1.0, 0.95, a);
        cairo_fill(cr);
    }

    /* coolant puffs */
    for (int i = 0; i < MAX_PUFFS; i++) {
        const puff *p = &puffs[i];
        if (!p->alive)
            continue;
        double sx, sy, u = p->age / p->life;
        proj(p->x, p->y, p->z, &sx, &sy);
        double a = clamp01(u * 5) * (1 - u) * (1 - u);
        blit(cr, puff_spr, sx, sy, p->size * (0.3 + 1.1 * sqrt(u)), 0.55 * a);
        blit_add(cr, puff_spr, sx, sy, p->size * (0.2 + 0.8 * sqrt(u)), 0.25 * a);
    }

    /* shuttles with short engine trails */
    for (int i = 0; i < MAX_SHUTTLES; i++) {
        shuttle *s = &shuttles[i];
        double x, y, a, th;
        if (!s->alive || !shuttle_pos(s, &x, &y, &a, &th))
            continue;
        if (a > 0.01) {
            if (!isnan(s->lx) && hypot(x - s->lx, y - s->ly) > 0.2) {
                double vx = x - s->lx, vy = y - s->ly, len = hypot(vx, vy);
                double tl = fmin(26, 3 + len * 5);
                vx /= len, vy /= len;
                cairo_pattern_t *g = cairo_pattern_create_linear(x, y, x - vx * tl, y - vy * tl);
                rgb c = s->arrive ? AMBER : CYAN;
                cairo_pattern_add_color_stop_rgba(g, 0, c.r, c.g, c.b, 0.55 * a * th);
                cairo_pattern_add_color_stop_rgba(g, 1, c.r, c.g, c.b, 0);
                cairo_move_to(cr, x, y);
                cairo_line_to(cr, x - vx * tl, y - vy * tl);
                cairo_set_source(cr, g);
                cairo_set_line_width(cr, 1.4);
                cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
                cairo_stroke(cr);
                cairo_pattern_destroy(g);
                /* hull: a small bright sliver along the motion */
                cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
                cairo_move_to(cr, x + vx * 2.6, y + vy * 2.6);
                cairo_line_to(cr, x - vx * 2.6, y - vy * 2.6);
                cairo_set_source_rgba(cr, 0.88, 0.9, 0.94, a);
                cairo_set_line_width(cr, 2.2);
                cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
                cairo_stroke(cr);
                cairo_set_line_cap(cr, CAIRO_LINE_CAP_BUTT);
                blit_add(cr, dot_spr[s->arrive ? DOT_AMBER : DOT_CYAN], x - vx * 3, y - vy * 3, 0.6, a * (0.3 + 0.7 * th));
            } else {
                cairo_arc(cr, x, y, 1.4, 0, 2 * M_PI);
                cairo_set_source_rgba(cr, 0.88, 0.9, 0.94, a);
                cairo_fill(cr);
            }
        }
        s->lx = x, s->ly = y;
    }

    /* mast tip beacon */
    {
        double bx, by;
        proj(0, MAST_LO - 12, 0, &bx, &by);
        blit_add(cr, dot_spr[DOT_RED], bx, by, 0.8, 0.15 + 0.85 * pow(0.5 + 0.5 * sin(t * 1.3 + 1), 10));
    }

    /* text: the current page, cross-fading into the next */
    update_hud(page, sh, t);
    cairo_set_source_surface(cr, hud_cache[page], 0, HUD_Y);
    cairo_paint_with_alpha(cr, 1 - fade);
    if (fade > 0) {
        update_hud(nx, sh, t);
        cairo_set_source_surface(cr, hud_cache[nx], 0, HUD_Y);
        cairo_paint_with_alpha(cr, fade);
    }
    update_top(sh);
    cairo_set_source_surface(cr, top_cache, 0, 0);
    cairo_paint(cr);
    for (int i = 0; i < N_PAGES; i++) {             /* page dots */
        double on = i == page ? 1 - fade : i == nx ? fade : 0;
        cairo_arc(cr, CX + (i - 1.5) * 12, 458, 2.4, 0, 2 * M_PI);
        cairo_set_source_rgba(cr, 0.75, 0.85, 1.0, 0.25 + 0.6 * on);
        cairo_fill(cr);
    }
}

/* ---------------------------------------------------------------- demo and showcase */

/* A smooth pseudo-random wiggle in -1..1 */
static double wig(double t, double seed)
{
    return 0.5 * sin(t * 1.3 + seed) + 0.3 * sin(t * 0.57 + seed * 2.1) + 0.2 * sin(t * 2.9 + seed * 0.7);
}

/*
 * Simulated sensors. level scales everything from quiet (0) to flat out (1); each
 * call builds fresh values from t and level, nothing is compounded.
 */
static void sim_sys(sys_stats *y, double t, double level, double io, double net, double gpu)
{
    y->n_cpu = 32;
    double tot = 0;
    for (int i = 0; i < 32; i++) {
        double w = 0.5 + 0.5 * wig(t * 0.7, i * 1.37);
        double spike = pow(0.5 + 0.5 * sin(t * 0.9 + i * 0.8), 8);
        y->cpu[i] = clamp01(0.02 + level * (0.35 + 0.6 * w) + 0.25 * spike * level + (level > 0.9 ? 0.3 : 0));
        tot += y->cpu[i];
    }
    y->cpu_total = tot / 32;
    y->cpu_temp = 38 + 54 * pow(level, 1.2) + 2 * wig(t * 0.3, 3);
    y->mem_total = 91;
    y->mem_used = 14 + 64 * level + 3 * wig(t * 0.2, 1);
    y->mem_cached = 4 + 6 * (1 - level);
    for (int k = 0; k < SYS_N_NVME; k++) {
        double b = io * (k == 0 ? 1 : k == 1 ? 0.5 : 0.25);
        y->nvme_rd[k] = b * 1.8e9 * pow(0.5 + 0.5 * sin(t * (0.5 + 0.2 * k) + k * 2), 3);
        y->nvme_wr[k] = b * 0.9e9 * pow(0.5 + 0.5 * sin(t * (0.37 + 0.15 * k) + k + 2), 3);
        y->nvme_temp[k] = 36 + 36 * io * (k == 0 ? 1 : 0.7) + wig(t * 0.2, k);
    }
    y->net_rx = net * 110e6 * (0.3 + 0.7 * pow(0.5 + 0.5 * sin(t * 0.35 + 0.5), 2)) + 4e3;
    y->net_tx = net * 30e6 * (0.2 + 0.8 * pow(0.5 + 0.5 * sin(t * 0.27 + 2.5), 2)) + 1.5e3;
    y->dimm_temp[0] = y->dimm_temp[1] = 40 + 12 * level;
    y->forks = 6 + 380 * pow(level, 1.5) * (0.6 + 0.4 * (0.5 + 0.5 * wig(t * 0.8, 9)));
    y->ctxt = 20000 + 900000 * level;
    y->tcp = 30 + 70 * net + 20 * level + 4 * wig(t * 0.1, 4);
    y->psi_cpu = level > 0.85 ? 90 * (level - 0.85) : 0;
    y->psi_io = io > 0.75 ? 60 * (io - 0.75) : 0;
    y->psi_mem = 1 + 4 * level;
    y->cpu_watts = 42 + 190 * pow(level, 1.1);
    y->uptime = 11 * 86400 + 7 * 3600 + 1260 + t;  /* day 12 */
    for (int k = 0; k < 2; k++)
        y->gpu_fan[k] = 30 + 70 * clamp01(gpu * (k ? 1.1 : 0.95));
}

static void sim_gpu(stats *s, double t, double g0, double g1)
{
    double g[2] = { g0, g1 };
    s->tok_s = 0;
    for (int k = 0; k < N_GPUS; k++) {
        double x = clamp01(g[k] + 0.05 * wig(t, k * 5) * (g[k] > 0.05));
        s->load[k] = x > 0.05 ? clamp01(0.3 + 0.7 * x) : 0;
        s->power[k] = 24 + 551 * x;
        s->temp[k] = (int)(32 + (k ? 38 : 30) * x);
        s->tok_port[k] = 0;
    }
    s->running = 0;
}

/* --demo: a slow wander through everything */
static void demo_all(stats *s, sys_stats *y, double t)
{
    double level = 0.5 + 0.5 * sin(t * 0.11);
    double io = pow(0.5 + 0.5 * sin(t * 0.17 + 1), 1.5);
    double net = pow(0.5 + 0.5 * sin(t * 0.13 + 2.2), 1.5);
    static stats demo_base;
    demo_poll(&demo_base, t);                       /* the shared GPU wander */
    *s = demo_base;
    s->power[0] = 24 + 550 * s->load[0];            /* scaled copy up to the 5090's limit */
    s->power[1] = 28 + 540 * s->load[1];
    s->tok_s = s->tok_port[0] = s->tok_port[1] = 0;
    s->running = 0;
    sim_sys(y, t, level, io, net, 0.5 * (s->load[0] + s->load[1]));
}

/*
 * --showcase: a scripted 36 s arc for filming: a quiet station, the CPU wakes up,
 * RAM fills, disks and network join, both reactors run flat out while everything
 * vents, then it all winds down again.
 */
#define SHOW_LEN 36.0
static void showcase_all(stats *s, sys_stats *y, double t)
{
    double u = fmod(t, SHOW_LEN);
    double up = smooth01((u - 3) / 9);              /* 3..12 CPU wakes */
    double io = smooth01((u - 8) / 6);              /* 8..14 disks */
    double net = smooth01((u - 10) / 5);
    double gpu0 = smooth01((u - 13) / 5), gpu1 = smooth01((u - 16) / 5);
    double peak = smooth01((u - 19) / 4);           /* 19..23 everything maxed */
    double down = smooth01((u - 27) / 7);           /* 27..34 wind down */
    double level = fmin(1, 0.03 + 0.6 * up + 0.4 * peak) * (1 - down);
    double g1 = gpu1 * (1 - smooth01((u - 25) / 7));
    sim_gpu(s, t, gpu0 * (1 - down), g1);
    sim_sys(y, t, level, io * (1 - down), net * (1 - down), fmax(gpu0 * (1 - down), g1));
    y->gpu_fan[0] = 30 + 70 * smooth01(gpu0 * (1 - smooth01((u - 28) / 6)));
    y->gpu_fan[1] = 30 + 70 * smooth01(gpu1 * (1 - smooth01((u - 30) / 5)));
    /* heat lags behind load on the way down */
    double lag = (1 - smooth01((u - 29) / 7));
    y->cpu_temp = 38 + 54 * pow(fmax(level, peak * lag), 1.2);
    for (int k = 0; k < SYS_N_NVME; k++)
        y->nvme_temp[k] = 36 + 36 * fmax(io * (1 - down), peak * lag) * (k == 0 ? 1 : 0.75);
    /* two short pressure alarms: the disks stall at 16..20 s, the CPU at 22.5..26 s */
    y->psi_io = 17 * smooth01(u - 16) * (1 - smooth01(u - 19));
    y->psi_cpu = 15 * smooth01(u - 22.5) * (1 - smooth01(u - 25));
}

/* ---------------------------------------------------------------- main */

static void ease_shown(shown_t *sh, const stats *s, const sys_stats *y, double dt, int snap)
{
    double k = snap ? 1 : 1 - exp(-dt * 3.0), kt = snap ? 1 : 1 - exp(-dt * 1.2);
    int nc = y->n_cpu < N_PODS ? y->n_cpu : N_PODS;
    for (int i = 0; i < N_PODS; i++)
        sh->pod[i] += ((i < nc ? y->cpu[i] : 0) - sh->pod[i]) * k;
    sh->cpu += (y->cpu_total - sh->cpu) * k;
    if (!isnan(y->cpu_temp))
        sh->cpu_temp += (y->cpu_temp - sh->cpu_temp) * kt;
    sh->mem_total = y->mem_total;
    sh->mem_used += (y->mem_used - sh->mem_used) * k;
    sh->mem_cached += (y->mem_cached - sh->mem_cached) * k;
    for (int i = 0; i < SYS_N_NVME; i++) {
        sh->rd[i] += (y->nvme_rd[i] - sh->rd[i]) * k;
        sh->wr[i] += (y->nvme_wr[i] - sh->wr[i]) * k;
        if (!isnan(y->nvme_temp[i]))
            sh->nvme_temp[i] += (y->nvme_temp[i] - sh->nvme_temp[i]) * kt;
    }
    sh->rx += (y->net_rx - sh->rx) * k;
    sh->tx += (y->net_tx - sh->tx) * k;
    for (int i = 0; i < N_GPUS; i++) {
        sh->power[i] += (s->power[i] - sh->power[i]) * k;
        sh->load[i] += (s->load[i] - sh->load[i]) * k;
        sh->temp[i] += (s->temp[i] - sh->temp[i]) * kt;
    }
    sh->tok += (s->tok_s - sh->tok) * k;
    for (int i = 0; i < N_GPUS && i < 2; i++)
        sh->fan[i] += (y->gpu_fan[i] - sh->fan[i]) * kt;
    sh->forks += (y->forks - sh->forks) * k;
    sh->tcp += (y->tcp - sh->tcp) * k;
    sh->psi_cpu += (y->psi_cpu - sh->psi_cpu) * k;
    sh->psi_io += (y->psi_io - sh->psi_io) * k;
    sh->psi_mem += (y->psi_mem - sh->psi_mem) * k;
    /* without RAPL (not root) the solar array follows an estimate from load */
    sh->watts_real = !isnan(y->cpu_watts);
    double cw = sh->watts_real ? y->cpu_watts : 45 + 175 * y->cpu_total;
    sh->cpu_watts += (cw - sh->cpu_watts) * k;
    sh->uptime = y->uptime;
}

int main(int argc, char **argv)
{
    cairo_surface_t *surf;
    cairo_t *cr;
    tjhandle tj;
    unsigned char *jpeg = NULL;
    stats s = { 0 };
    sys_stats y = { 0 };
    shown_t sh = { 0 };
    double last, next_poll = 0, t0;
    int fd = -1, bench = 0, showcase = 0, active = 0, first = 1;

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
    build_sprites();
    build_world();
    build_ring_sprites();

    surf = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cr = cairo_create(surf);
    tj = tj3Init(TJINIT_COMPRESS);
    tj3Set(tj, TJPARAM_QUALITY, JPEG_QUALITY);
    tj3Set(tj, TJPARAM_SUBSAMP, TJSAMP_420);

    if (bench) {
        /* showcase moments: quiet, waking, flat out */
        struct { double at; const char *png; } scenes[] = {
            { 22.0, "station_preview.png" }, { 11.0, "station_mid.png" }, { 1.5, "station_idle.png" },
        };
        for (int k = 0; k < 3; k++) {
            srand(5 + k);
            memset(shuttles, 0, sizeof(shuttles));
            memset(puffs, 0, sizeof(puffs));
            memset(packets, 0, sizeof(packets));
            memset(&sh, 0, sizeof(sh));
            rot = 0;
            int nf = 8 * FPS_SHOW;
            size_t len = 0;
            double b0 = 0;
            for (int i = 0; i < nf; i++) {
                double t = scenes[k].at - 8.0 + i / (double)FPS_SHOW;
                if (i == nf - 100)
                    b0 = now_s();
                showcase_all(&s, &y, t);
                ease_shown(&sh, &s, &y, 1.0 / FPS_SHOW, i == 0);
                active = simulate(&sh, 1.0 / FPS_SHOW);
                render(cr, &sh, t, k == 0 ? 1 : 0, 0, 0);
                cairo_surface_flush(surf);
                tj3Compress8(tj, cairo_image_surface_get_data(surf), SIZE, cairo_image_surface_get_stride(surf),
                             SIZE, TJPF_BGRX, &jpeg, &len);
            }
            printf("%s: %.2f ms/frame, jpeg %zu bytes, %d moving things\n", scenes[k].png,
                   (now_s() - b0) * 1000 / 100, len, active);
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
            showcase_all(&s, &y, t - t0);
        } else if (t >= next_poll) {
            next_poll = t + 0.5;                    /* sensors at 2 Hz */
            if (demo) {
                demo_all(&s, &y, t - t0);
            } else {
                gpus_poll(&s);
                gpu_fans_poll(&y);
                sys_poll(&y, t);
                if (gpu_source)
                    gpu_rate_poll(&s);
                else
                    vllm_poll(&s, t);
            }
        }
        ease_shown(&sh, &s, &y, dt, first);
        first = 0;

        active = simulate(&sh, dt);
        double el = t - t0;
        int page = (int)(el / PAGE_SECS) % N_PAGES, nx = (page + 1) % N_PAGES;
        double fade = smooth01((fmod(el, PAGE_SECS) - (PAGE_SECS - 0.8)) / 0.8);
        if (showcase) {
            /* pages follow the story: CPU, I/O, GPU, RAM, CPU again */
            static const double at[] = { 0, 9.5, 15.5, 24, 29.5, SHOW_LEN };
            static const int pg[] = { 0, 2, 1, 3, 0 };
            double u = fmod(el, SHOW_LEN);
            int i = 0;
            while (i < 4 && u >= at[i + 1])
                i++;
            page = pg[i];
            nx = pg[(i + 1) % 5];
            fade = smooth01((u - (at[i + 1] - 0.8)) / 0.8);
        }
        render(cr, &sh, el, page, nx, fade);
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

        /* quiet station: CPU, GPUs, disks and network near idle */
        double io = sh.rx + sh.tx;
        for (int k = 0; k < SYS_N_NVME; k++)
            io += sh.rd[k] + sh.wr[k];
        int idle = !showcase && sh.cpu < 0.08 && sh.power[0] < 90 && sh.power[1] < 90 && io < 2e6;
        (void)active;
        double spare = 1.0 / (showcase ? FPS_SHOW : idle ? FPS_IDLE : FPS_BUSY) - (now_s() - t);
        if (spare > 0) {
            struct timespec ts = { 0, (long)(spare * 1e9) };
            nanosleep(&ts, NULL);
        }
    }

    tj3Free(jpeg);
    tj3Destroy(tj);
    cairo_destroy(cr);
    cairo_surface_destroy(surf);
    cairo_surface_t *all[] = { bg_layer, wall_layer, mast_lo_layer, hub_layer, mast_hi_layer, solar_lit,
                               pod_glow, streak_spr[0], streak_spr[1], core_dark[0], core_dark[1], core_hot[0],
                               core_hot[1], react_glow[0], react_glow[1], puff_spr, dot_spr[0], dot_spr[1],
                               dot_spr[2], dot_spr[3], sat_spr, hud_cache[0], hud_cache[1], hud_cache[2],
                               hud_cache[3], top_cache };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++)
        cairo_surface_destroy(all[i]);
    if (nvml_ok)
        nvmlShutdown();
    if (fd >= 0)
        close(fd);
    return 0;
}
