/*
 * hamsters: the computer is secretly powered by hamsters, on the iCUE LINK AIO pump LCD.
 *
 * A cutaway of the hamster power plant under your desk. Sixteen hamsters on two
 * shelves of wheels are the sixteen CPU cores: each runs as fast as its core is busy,
 * turns red and sweats as the CPU heats up, and curls up for a nap (zzz) when its
 * core is idle. Two big chonky hamsters in hard hats on giant wheels are the two
 * GPUs (blue GPU 0, orange GPU 1), spinning with GPU load and throwing sparks at high
 * power. The food bowl is RAM: it drains as memory fills and a caretaker tops it up
 * when memory is freed. Three burrows are the three NVMe drives: hamsters with
 * stuffed cheeks carry seeds in (writes) and pop out spitting seeds (reads). A
 * hamster on a red telephone is the network. Context switches make neighbours hop
 * out and swap wheels, forks send baby hamsters along the shelves, major page faults
 * make a hamster lose its footing and loop the loop. Rim gauges show the CPU clock
 * ("RPM", left) and package watts (hamster-power, right).
 * The hamster sprites and the burrow painting were generated with an image model
 * (assets/hamsters/); wheels, bowl, burrow doors and effects are drawn with cairo.
 * Driven by system sensors (/proc, hwmon) and NVML; vLLM is not needed.
 * Run with --demo to simulate data, --showcase for a scripted 36 s arc from quiet to
 * maxed out and back, --bench to write preview PNGs.
 */
#define _GNU_SOURCE
#include <cairo/cairo.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/hidraw.h>
#include <math.h>
#include <nvml.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <turbojpeg.h>
#include <unistd.h>

#define SIZE            480
#define REPORT_SIZE     1024
#define HEADER_SIZE     8
#define CHUNK_SIZE      (REPORT_SIZE - HEADER_SIZE)
#define JPEG_QUALITY    85
#define N_GPUS          2

static const char  *gpu_bus[N_GPUS]  = { "00000000:01:00.0", "00000000:03:00.0" };  /* ZOTAC, TUF */

typedef struct { double r, g, b; } rgb;

__attribute__((unused)) static const rgb BLUE   = { 61 / 255.0, 174 / 255.0, 233 / 255.0 };
__attribute__((unused)) static const rgb ORANGE = { 233 / 255.0, 120 / 255.0, 61 / 255.0 };
__attribute__((unused)) static const rgb WHITE  = { 240 / 255.0, 240 / 255.0, 245 / 255.0 };

typedef struct {
    double load[N_GPUS];        /* 0..1 */
    double power[N_GPUS];       /* W */
    int    temp[N_GPUS];        /* C */
} stats;

static volatile sig_atomic_t stop;
static int demo;

/*
 * GPU activity, as in the other displays' GPU mode: half utilisation, half power draw
 * between GPU_IDLE_W and GPU_MAX_W (utilisation alone can sit at 100% while the card
 * is barely working; the watts show how hard it really is). This display is always
 * driven by GPU activity, so --gpu-load / LLM_REACTOR_SOURCE=gpu are accepted and
 * change nothing.
 */
#define GPU_IDLE_W      40.0        /* board power at idle */
#define GPU_MAX_W       575.0       /* board power limit */

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

/* ---------------------------------------------------------------- GPUs */

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

/* GPU activity 0..1 (see GPU_IDLE_W): below 3% counts as idle */
static double gpu_activity(const stats *s, int i)
{
    double pw = clamp01((s->power[i] - GPU_IDLE_W) / (GPU_MAX_W - GPU_IDLE_W));
    double a = 0.5 * clamp01(s->load[i]) + 0.5 * pw;
    return a < 0.03 ? 0 : a;
}

/* ---- system sensors ---- */

/*
 * Everything else comes from the kernel: per-thread CPU load, context switches,
 * interrupts and forks (/proc/stat), per-thread clocks (cpufreq), CPU package watts
 * (RAPL energy counter; root only, estimated from load and clock when unreadable),
 * CPU and board temperatures (hwmon, found by name, never by number), RAM
 * (/proc/meminfo), page faults (/proc/vmstat), pressure stall (/proc/pressure),
 * NVMe throughput (/proc/diskstats) and network (/proc/net/dev). sys_init() finds
 * the files once; sys_poll() reads them (cheap, meant for ~2 Hz) and differences the
 * counters into rates. Missing sensors just stay at 0.
 */
#define SYS_MAX_THREADS 64
#define N_CORES         16
#define N_NVME          3

typedef struct {
    int    n_threads;
    double thread_load[SYS_MAX_THREADS];    /* 0..1 per hardware thread */
    double core_load[N_CORES];              /* 0..1 per physical core (SMT siblings averaged) */
    double cpu_load;                        /* 0..1 whole package */
    double cpu_temp;                        /* k10temp Tctl, C */
    double cpu_pkg_temp, mb_temp, vrm_temp; /* asusec, C */
    double cpu_mhz, cpu_mhz_max;            /* average and fastest thread clock */
    double pkg_watts;                       /* CPU package power */
    int    pkg_watts_measured;              /* 1: RAPL, 0: estimated from load and clock */
    double ctxt_s, intr_s, forks_s;         /* context switches, interrupts, new processes per second */
    int    procs_running;
    double pgfault_s, pgmajfault_s;         /* page faults per second, major = had to go to disk/swap */
    double psi_cpu, psi_io, psi_mem;        /* pressure stall "some avg10", % */
    double ram_total, ram_used, ram_cached; /* GB (used = total - available) */
    double dimm_temp[2];                    /* spd5118, C */
    double nvme_temp[N_NVME];               /* C */
    double nvme_rd[N_NVME], nvme_wr[N_NVME];/* bytes/s, nvme0n1..nvme2n1 */
    double net_rx, net_tx;                  /* enp12s0, bytes/s */
    double ts_rx, ts_tx;                    /* tailscale0, bytes/s */
} sys_stats;

static const char *NET_IF = "enp12s0", *TS_IF = "tailscale0";

static char   k10_temp_path[300], asus_pkg_path[300], asus_mb_path[300], asus_vrm_path[300];
static int    rapl_ok = 1;                  /* cleared for good on EACCES */
static char   dimm_path[2][300], nvme_path[N_NVME][300];
static int    core_of[SYS_MAX_THREADS];     /* hardware thread -> core index */

static int read_text(const char *path, char *buf, int cap)
{
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return -1;
    ssize_t n = read(fd, buf, cap - 1);
    close(fd);
    if (n < 0)
        return -1;
    buf[n] = 0;
    return (int)n;
}

static double read_milli(const char *path)
{
    char buf[32];
    if (!*path || read_text(path, buf, sizeof(buf)) <= 0)
        return 0;
    return strtod(buf, NULL) / 1000.0;
}

static void sys_init(void)
{
    char path[300], buf[128];
    int n_dimm = 0;
    DIR *d = opendir("/sys/class/hwmon");
    struct dirent *e;

    while (d && (e = readdir(d))) {
        if (strncmp(e->d_name, "hwmon", 5))
            continue;
        snprintf(path, sizeof(path), "/sys/class/hwmon/%s/name", e->d_name);
        if (read_text(path, buf, sizeof(buf)) <= 0)
            continue;
        buf[strcspn(buf, "\n")] = 0;
        if (!strcmp(buf, "k10temp")) {
            snprintf(k10_temp_path, sizeof(k10_temp_path), "/sys/class/hwmon/%s/temp1_input", e->d_name);
        } else if (!strcmp(buf, "spd5118") && n_dimm < 2) {
            snprintf(dimm_path[n_dimm++], sizeof(dimm_path[0]), "/sys/class/hwmon/%s/temp1_input", e->d_name);
        } else if (!strcmp(buf, "nvme")) {
            /* which drive: the hwmon's device is nvmeN */
            char link[300], target[300];
            snprintf(link, sizeof(link), "/sys/class/hwmon/%s/device", e->d_name);
            ssize_t n = readlink(link, target, sizeof(target) - 1);
            int idx = -1;
            if (n > 0) {
                target[n] = 0;
                const char *b = strrchr(target, '/');
                b = b ? b + 1 : target;
                if (!strncmp(b, "nvme", 4))
                    idx = atoi(b + 4);
            }
            if (idx < 0 || idx >= N_NVME)
                for (idx = 0; idx < N_NVME && nvme_path[idx][0]; idx++)
                    ;
            if (idx < N_NVME)
                snprintf(nvme_path[idx], sizeof(nvme_path[0]), "/sys/class/hwmon/%s/temp1_input", e->d_name);
        } else if (!strcmp(buf, "asusec")) {
            for (int i = 1; i <= 8; i++) {
                snprintf(path, sizeof(path), "/sys/class/hwmon/%s/temp%d_label", e->d_name, i);
                if (read_text(path, buf, sizeof(buf)) <= 0)
                    continue;
                buf[strcspn(buf, "\n")] = 0;
                char *dst = !strcmp(buf, "CPU Package") ? asus_pkg_path : !strcmp(buf, "Motherboard") ? asus_mb_path
                          : !strcmp(buf, "VRM") ? asus_vrm_path : NULL;
                if (dst)
                    snprintf(dst, 300, "/sys/class/hwmon/%s/temp%d_input", e->d_name, i);
            }
        }
    }
    if (d)
        closedir(d);

    /* Physical cores: threads sharing thread_siblings_list's first entry are one core */
    int first_of_core[N_CORES * 4], n_cores = 0;
    for (int t = 0; t < SYS_MAX_THREADS; t++) {
        int first = t % N_CORES;
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", t);
        if (read_text(path, buf, sizeof(buf)) > 0)
            first = atoi(buf);
        int c;
        for (c = 0; c < n_cores && first_of_core[c] != first; c++)
            ;
        if (c == n_cores && n_cores < N_CORES * 4)
            first_of_core[n_cores++] = first;
        core_of[t] = c % N_CORES;
    }
}

static void sys_poll(sys_stats *s, double t)
{
    static unsigned long long last_busy[SYS_MAX_THREADS], last_total[SYS_MAX_THREADS];
    static unsigned long long last_rd[N_NVME], last_wr[N_NVME], last_net[4];
    static unsigned long long last_ctxt, last_intr, last_forks, last_pgf, last_majf, last_energy;
    static double last_t;
    static int have;
    double dt = t - last_t;
    char line[8192];
    FILE *f;

    /* CPU: per-thread busy share since the last poll */
    if ((f = fopen("/proc/stat", "r"))) {
        double core_sum[N_CORES] = { 0 };
        int core_n[N_CORES] = { 0 };
        unsigned long long all_busy = 0, all_total = 0;
        int n = 0;
        while (fgets(line, sizeof(line), f)) {
            unsigned long long v[8] = { 0 }, x;
            int id;
            if (sscanf(line, "ctxt %llu", &x) == 1) {
                if (have && dt > 0 && x >= last_ctxt)
                    s->ctxt_s = (x - last_ctxt) / dt;
                last_ctxt = x;
            } else if (sscanf(line, "intr %llu", &x) == 1) {  /* first number is the total */
                if (have && dt > 0 && x >= last_intr)
                    s->intr_s = (x - last_intr) / dt;
                last_intr = x;
            } else if (sscanf(line, "processes %llu", &x) == 1) {
                if (have && dt > 0 && x >= last_forks)
                    s->forks_s = (x - last_forks) / dt;
                last_forks = x;
            } else if (sscanf(line, "procs_running %llu", &x) == 1) {
                s->procs_running = (int)x;
            }
            if (strncmp(line, "cpu", 3) || line[3] == ' ')
                continue;
            if (sscanf(line + 3, "%d %llu %llu %llu %llu %llu %llu %llu %llu", &id, &v[0], &v[1], &v[2], &v[3],
                       &v[4], &v[5], &v[6], &v[7]) < 9 || id < 0 || id >= SYS_MAX_THREADS)
                continue;
            unsigned long long total = 0, idle = v[3] + v[4];
            for (int k = 0; k < 8; k++)
                total += v[k];
            unsigned long long busy = total - idle;
            if (have && total > last_total[id]) {
                double l = (double)(busy - last_busy[id]) / (double)(total - last_total[id]);
                s->thread_load[id] = clamp01(l);
                all_busy += busy - last_busy[id];
                all_total += total - last_total[id];
            }
            last_busy[id] = busy;
            last_total[id] = total;
            core_sum[core_of[id]] += s->thread_load[id];
            core_n[core_of[id]]++;
            if (id + 1 > n)
                n = id + 1;
        }
        fclose(f);
        s->n_threads = n;
        for (int c = 0; c < N_CORES; c++)
            s->core_load[c] = core_n[c] ? core_sum[c] / core_n[c] : 0;
        if (all_total)
            s->cpu_load = (double)all_busy / all_total;
    }

    /* Clocks: every thread's current frequency */
    double mhz_sum = 0, mhz_max = 0;
    for (int i = 0; i < s->n_threads; i++) {
        char p[96], b[32];
        snprintf(p, sizeof(p), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", i);
        if (read_text(p, b, sizeof(b)) > 0) {
            double m = strtod(b, NULL) / 1000.0;
            mhz_sum += m;
            mhz_max = fmax(mhz_max, m);
        }
    }
    s->cpu_mhz = s->n_threads ? mhz_sum / s->n_threads : 0;
    s->cpu_mhz_max = mhz_max;

    /* Package power: RAPL energy counter (root only), else a rough estimate */
    if (rapl_ok) {
        char b[48];
        int fd = open("/sys/class/powercap/intel-rapl:0/energy_uj", O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            rapl_ok = 0;                                /* EACCES or missing: estimate from now on */
        } else {
            ssize_t n = read(fd, b, sizeof(b) - 1);
            close(fd);
            if (n > 0) {
                b[n] = 0;
                unsigned long long e = strtoull(b, NULL, 10);
                if (have && dt > 0 && e >= last_energy && last_energy)
                    s->pkg_watts = (e - last_energy) / 1e6 / dt;
                last_energy = e;
                s->pkg_watts_measured = 1;
            }
        }
    }
    if (!rapl_ok) {
        s->pkg_watts = 28 + 190 * s->cpu_load * (0.55 + 0.45 * clamp01(s->cpu_mhz / 5500));
        s->pkg_watts_measured = 0;
    }

    /* Page faults */
    if ((f = fopen("/proc/vmstat", "r"))) {
        unsigned long long x;
        while (fgets(line, sizeof(line), f)) {
            if (sscanf(line, "pgfault %llu", &x) == 1) {
                if (have && dt > 0 && x >= last_pgf)
                    s->pgfault_s = (x - last_pgf) / dt;
                last_pgf = x;
            } else if (sscanf(line, "pgmajfault %llu", &x) == 1) {
                if (have && dt > 0 && x >= last_majf)
                    s->pgmajfault_s = (x - last_majf) / dt;
                last_majf = x;
                break;                                  /* comes after pgfault */
            }
        }
        fclose(f);
    }

    /* Pressure stall information: share of time something was waiting */
    static const char *psi_files[3] = { "/proc/pressure/cpu", "/proc/pressure/io", "/proc/pressure/memory" };
    double *psi[3] = { &s->psi_cpu, &s->psi_io, &s->psi_mem };
    for (int i = 0; i < 3; i++) {
        char b[256];
        const char *a;
        if (read_text(psi_files[i], b, sizeof(b)) > 0 && (a = strstr(b, "some avg10=")))
            *psi[i] = strtod(a + 11, NULL);
    }

    /* Temperatures */
    s->cpu_temp = read_milli(k10_temp_path);
    s->cpu_pkg_temp = read_milli(asus_pkg_path);
    s->mb_temp = read_milli(asus_mb_path);
    s->vrm_temp = read_milli(asus_vrm_path);
    for (int i = 0; i < 2; i++)
        s->dimm_temp[i] = read_milli(dimm_path[i]);
    for (int i = 0; i < N_NVME; i++)
        s->nvme_temp[i] = read_milli(nvme_path[i]);

    /* RAM */
    if ((f = fopen("/proc/meminfo", "r"))) {
        double total = 0, avail = 0, cached = 0, v;
        while (fgets(line, sizeof(line), f)) {
            if (sscanf(line, "MemTotal: %lf", &v) == 1)
                total = v;
            else if (sscanf(line, "MemAvailable: %lf", &v) == 1)
                avail = v;
            else if (sscanf(line, "Cached: %lf", &v) == 1) {
                cached = v;
                break;                                  /* comes after the other two */
            }
        }
        fclose(f);
        s->ram_total = total / 1048576.0;
        s->ram_used = (total - avail) / 1048576.0;
        s->ram_cached = cached / 1048576.0;
    }

    /* NVMe throughput: sectors read (field 6) and written (field 10), 512 bytes each */
    if ((f = fopen("/proc/diskstats", "r"))) {
        while (fgets(line, sizeof(line), f)) {
            char name[32];
            unsigned long long rd, wr;
            if (sscanf(line, "%*u %*u %31s %*u %*u %llu %*u %*u %*u %llu", name, &rd, &wr) != 3)
                continue;
            int i;
            if (strncmp(name, "nvme", 4) || strlen(name) != 7 || strcmp(name + 5, "n1") ||
                (i = name[4] - '0') < 0 || i >= N_NVME)
                continue;
            if (have && dt > 0) {
                s->nvme_rd[i] = rd >= last_rd[i] ? (rd - last_rd[i]) * 512.0 / dt : 0;
                s->nvme_wr[i] = wr >= last_wr[i] ? (wr - last_wr[i]) * 512.0 / dt : 0;
            }
            last_rd[i] = rd;
            last_wr[i] = wr;
        }
        fclose(f);
    }

    /* Network */
    if ((f = fopen("/proc/net/dev", "r"))) {
        while (fgets(line, sizeof(line), f)) {
            char *colon = strchr(line, ':'), *name = line;
            unsigned long long rx, tx;
            if (!colon)
                continue;
            *colon = 0;
            while (*name == ' ')
                name++;
            int k = !strcmp(name, NET_IF) ? 0 : !strcmp(name, TS_IF) ? 2 : -1;
            if (k < 0 || sscanf(colon + 1, "%llu %*u %*u %*u %*u %*u %*u %*u %llu", &rx, &tx) != 2)
                continue;
            if (have && dt > 0) {
                double r = rx >= last_net[k] ? (rx - last_net[k]) / dt : 0;
                double w = tx >= last_net[k + 1] ? (tx - last_net[k + 1]) / dt : 0;
                if (k == 0)
                    s->net_rx = r, s->net_tx = w;
                else
                    s->ts_rx = r, s->ts_tx = w;
            }
            last_net[k] = rx;
            last_net[k + 1] = tx;
        }
        fclose(f);
    }
    last_t = t;
    have = 1;
}

/* ---- end system sensors ---- */

/* ---------------------------------------------------------------- simulated data */

static double frand(void) { return rand() / (double)RAND_MAX; }

static void sys_fill_derived(sys_stats *s)
{
    double sum = 0;
    s->n_threads = 32;
    for (int c = 0; c < N_CORES; c++) {
        s->thread_load[c] = s->core_load[c];
        s->thread_load[c + N_CORES] = s->core_load[c] * 0.8;
        sum += s->core_load[c];
    }
    s->cpu_load = sum / N_CORES;
    s->ram_total = 91.9;
    s->cpu_pkg_temp = s->cpu_temp + 1;
    s->mb_temp = 34 + s->cpu_load * 6;
    s->vrm_temp = 45 + s->cpu_load * 25;
    s->dimm_temp[0] = s->dimm_temp[1] = 42 + s->ram_used / s->ram_total * 10;
    for (int i = 0; i < N_NVME; i++)
        s->nvme_temp[i] = 40 + clamp01((s->nvme_rd[i] + s->nvme_wr[i]) / 3e9) * 25;
}

/* --demo: everything wanders through idle, busy and flat out, each on its own rhythm */
static void demo_poll(stats *g, sys_stats *s, double t)
{
    double busy = clamp01(0.5 + 0.62 * sin(t * 0.16) + 0.25 * sin(t * 0.43));
    for (int c = 0; c < N_CORES; c++) {
        double own = 0.5 + 0.5 * sin(t * (0.3 + 0.07 * c) + c * 1.7);
        double l = busy * (0.35 + 0.65 * own) + (frand() - 0.5) * 0.08;
        s->core_load[c] = clamp01(l < 0.08 ? 0.01 : l);
    }
    s->cpu_temp = 45 + 48 * pow(busy, 1.3);
    s->ram_used = 16 + 70 * clamp01(0.5 + 0.5 * sin(t * 0.09 - 1));
    for (int i = 0; i < N_NVME; i++) {
        double r = 0.5 + 0.5 * sin(t * (0.21 + 0.1 * i) + i * 2.1);
        double w = 0.5 + 0.5 * sin(t * (0.17 + 0.08 * i) + i * 1.3 + 2);
        s->nvme_rd[i] = r > 0.55 ? pow(10, 5 + 4.3 * (r - 0.55) / 0.45) : 0;
        s->nvme_wr[i] = w > 0.6 ? pow(10, 5 + 4 * (w - 0.6) / 0.4) : 0;
    }
    double net = 0.5 + 0.5 * sin(t * 0.27 + 0.5);
    s->net_rx = pow(10, 3 + 5 * net);
    s->net_tx = pow(10, 3 + 4 * (0.5 + 0.5 * sin(t * 0.33 + 2)));
    s->ts_rx = s->ts_tx = 2000;
    for (int i = 0; i < N_GPUS; i++) {
        double a = clamp01(0.5 + 0.7 * sin(t * (0.12 + 0.05 * i) + i * 2.4));
        g->load[i] = a < 0.1 ? 0 : clamp01(a + (frand() - 0.5) * 0.06);
        g->power[i] = (i ? 32 : 20) + 540 * pow(a, 1.2);
        g->temp[i] = (int)(36 + 42 * a);
    }
    s->cpu_mhz = 1100 + 4300 * pow(busy, 0.45);
    s->pkg_watts = 32 + 190 * busy;
    s->pkg_watts_measured = 1;
    s->ctxt_s = pow(10, 4.4 + 2.0 * busy);
    s->forks_s = pow(10, 0.5 + 3.2 * clamp01(0.5 + 0.6 * sin(t * 0.23 + 1)));
    double m = 0.5 + 0.5 * sin(t * 0.19 + 2);
    s->pgmajfault_s = m > 0.65 ? pow(10, 2 + 2.3 * (m - 0.65) / 0.35) : 5;
    s->psi_cpu = 30 * pow(busy, 3);
    sys_fill_derived(s);
}

/*
 * --showcase: a scripted 36 s arc for filming or GIFs. Everyone asleep; a download
 * wakes the network and a few cores; a build heats everything up; GPUs join and it
 * all goes flat out (bowl nearly empty, sparks, sweat); memory is freed and the
 * caretaker refills the bowl; everything winds down and goes back to sleep.
 */
#define SHOWCASE_LEN 36.0

static double ramp(double u, double a, double b) { return clamp01((u - a) / (b - a)); }

static void showcase_poll(stats *g, sys_stats *s, double t)
{
    double u = fmod(t, SHOWCASE_LEN);
    /* overall CPU intensity */
    double cpu = u < 4 ? 0 : u < 9 ? 0.35 * ramp(u, 4, 7) : u < 15 ? 0.35 + 0.55 * ramp(u, 9, 13)
               : u < 24 ? 1.0 : u < 29 ? 1.0 - 0.8 * ramp(u, 24, 27) : 0.2 * (1 - ramp(u, 29, 31));
    for (int c = 0; c < N_CORES; c++) {
        double jitter = 0.5 + 0.5 * sin(u * (1.1 + 0.13 * c) + c * 2.3);
        double wake = (c * 7 % N_CORES) / (double)N_CORES;       /* order they wake in */
        double l;
        if (cpu < 0.4)
            l = wake < cpu * 2.2 ? 0.25 + 0.4 * jitter : 0.01;
        else
            l = cpu * (0.85 + 0.15 * jitter);
        if (u < 4 && c == 5 && u > 2)                              /* one early riser */
            l = 0.2;
        s->core_load[c] = clamp01(l);
    }
    s->cpu_temp = 44 + 51 * clamp01(0.1 * ramp(u, 4, 9) + 0.9 * pow(ramp(u, 9, 18), 1.2))
                  * (1 - 0.8 * ramp(u, 24, 31));
    s->ram_used = 18 + 22 * ramp(u, 5, 9) + 30 * ramp(u, 10, 14) + 18 * ramp(u, 16, 19)
                  - 63 * ramp(u, 24.5, 26.5);
    for (int i = 0; i < N_NVME; i++) {
        s->nvme_rd[i] = s->nvme_wr[i] = 0;
    }
    if (u > 5 && u < 10)
        s->nvme_rd[0] = 4e8;                                       /* loading the project */
    if (u > 10 && u < 24) {
        s->nvme_wr[1] = 9e8 * (0.6 + 0.4 * sin(u * 2));            /* build output */
        s->nvme_rd[0] = 2e8;
    }
    if (u > 16 && u < 24) {
        s->nvme_wr[0] = 5e8;
        s->nvme_wr[2] = 2e9;
        s->nvme_rd[2] = 1.5e9;
        s->nvme_rd[1] = 6e8;
    }
    if (u > 24 && u < 27)
        s->nvme_wr[2] = 3e8;                                       /* saving the results */
    s->net_rx = u < 4 ? 3e3 : u < 10 ? 8e7 * ramp(u, 4, 5) : u < 24 ? 1e6 + 3e7 * ramp(u, 16, 18) : u < 28 ? 2e5 : 3e3;
    s->net_tx = u < 4 ? 1e3 : u < 16 ? 5e4 : u < 24 ? 6e7 * ramp(u, 17, 19) : 4e3;
    s->ts_rx = s->ts_tx = 1000;
    for (int i = 0; i < N_GPUS; i++) {
        double a = i == 0 ? ramp(u, 13, 16) : ramp(u, 16, 18);
        a *= 1 - ramp(u, 25, 29 - i);
        a = a > 0.02 ? clamp01(a * (0.93 + 0.07 * sin(u * 3 + i))) : 0;
        g->load[i] = a;
        g->power[i] = (i ? 32 : 20) + 550 * a;
        g->temp[i] = (int)(36 + 44 * a);
    }
    /* clocks and package power follow the CPU; the build forks like mad; a burst of
       major faults when memory runs short (and while the project loads) */
    s->cpu_mhz = 900 + 4300 * pow(cpu, 0.5) - 250 * ramp(s->cpu_temp, 88, 95);
    s->pkg_watts = 26 + 200 * cpu * (0.9 + 0.1 * sin(u * 2.3));
    s->pkg_watts_measured = 1;
    s->ctxt_s = pow(10, 4.1 + 2.2 * cpu);
    s->forks_s = u < 4 ? 3 : u < 9 ? 40 : u < 24 ? pow(10, 1.6 + 1.9 * ramp(u, 9, 11)) : u < 28 ? 30 : 3;
    s->pgmajfault_s = (u > 5 && u < 8) ? 400 : (u > 18 && u < 24) ? 9000 : 10;
    s->psi_cpu = 40 * pow(cpu, 4);
    sys_fill_derived(s);
}

/* ---------------------------------------------------------------- scene */

#define FPS_BUSY        24
#define FPS_IDLE        15
#define POLL_S          0.5         /* sensor reads, 2 Hz */

/* CPU wheels: two shelves of eight */
#define CW_R            22.0
#define CW_PITCH        47.0
static const double CW_ROW_Y[2] = { 124, 181 };
#define SHELF_DY        25.0        /* shelf top below a wheel's hub */
/* GPU wheels */
#define GW_R            64.0
static const double GW_X[N_GPUS] = { 121, 359 };
#define GW_Y            298.0
#define FLOOR_Y         404.0
/* Bowl (RAM) */
#define BOWL_X          240.0
#define BOWL_RIM_Y      370.0
#define BOWL_RX         46.0
#define BOWL_RY         9.0
#define BOWL_BOT_Y      403.0
/* Burrows (NVMe) */
static const double HOLE_X[N_NVME] = { 170, 240, 310 };
#define HOLE_FLOOR_Y    454.0
#define HOLE_W          40.0
#define HOLE_H          38.0
/* Phone (network) */
#define PHONE_X         240.0
#define PHONE_BOT_Y     316.0
#define PHONE_H         70.0

#define MAX_PARTS       900
#define MAX_CRITTERS    40

typedef enum { P_SWEAT, P_SPARK, P_SEED, P_Z, P_BANG, P_STAR } ptype;
typedef struct {
    double x, y, vx, vy, age, life, rot, vrot, size;
    ptype  type;
    int    alive;
} particle;

/* burrow visitors: carriers walk in with a seed (write), spitters pop out (read) */
typedef struct { double x, age, life; int hole, spit, alive, spat; } critter;

typedef struct {
    double load, angle, phase, sleep_t, heat;
    int    asleep;
    double z_acc, sweat_acc;
    double tumble;                  /* >0: lost its footing, going round with the wheel */
    int    away;                    /* jumping to the next wheel (context switch) */
} runner;

/* context switch: two neighbours tag out and swap wheels */
typedef struct { int a, b, alive; double age; } swap_t;
/* fork: a baby hamster scurries along a shelf */
typedef struct { int row, alive; double x, dir, speed, age, life, phase; } baby_t;
#define MAX_SWAPS   3
#define MAX_BABIES  12
#define SWAP_TIME   0.55
#define TUMBLE_TIME 0.9

/* Everything on screen, eased toward the latest readings */
typedef struct {
    double core[N_CORES];
    double cpu_load, cpu_temp;
    double gpu_act[N_GPUS], gpu_power[N_GPUS], gpu_temp[N_GPUS];
    double ram_used, ram_total, ram_free_frac;
    double io_rd[N_NVME], io_wr[N_NVME];        /* log-scaled 0..1 */
    double net_rx, net_tx;                      /* log-scaled 0..1 */
    double power;                               /* 0..1 how hard the plant works, for the lamps */
    double cpu_ghz, pkg_watts;                  /* the RPM gauge and the hamster-power meter */
    int    pkg_measured;
    double ctxt, forks, majf;                   /* log-scaled 0..1 event rates */
} view_t;

static particle parts[MAX_PARTS];
static critter  critters[MAX_CRITTERS];
static runner   cpu_run[N_CORES], gpu_run[N_GPUS];
static double   io_acc[N_NVME][2], spark_acc[N_GPUS], keeper_t, keeper_a, pour_acc, ram_rise;
static double   swap_acc, baby_acc, tumble_acc;
static swap_t   swaps[MAX_SWAPS];
static baby_t   babies[MAX_BABIES];
static double   net_phase[2], phone_jit;
static int      keeper_on;

static cairo_surface_t *bg_cache, *hud_cache, *bowl_front, *seed_heap, *z_spr[3], *bang_spr;
static cairo_surface_t *baby_spr[4], *lamp_glow, *gpu_glow[N_GPUS];
static cairo_surface_t *run_spr[4], *run_hot[4], *sleep_spr, *gpu_spr[N_GPUS][2], *gpu_hot[N_GPUS][2];
static cairo_surface_t *gpu_sleep_spr[N_GPUS], *carry_spr, *spit_spr, *phone_spr, *keeper_spr;
static char hud_key[160];

/* Assets live next to the binary (assets/hamsters/), or in ./assets/hamsters when run from the repo */
static cairo_surface_t *load_asset(const char *name)
{
    char exe[512], path[1024];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n > 0) {
        exe[n] = 0;
        char *slash = strrchr(exe, '/');
        if (slash)
            *slash = 0;
        snprintf(path, sizeof(path), "%s/assets/hamsters/%s", exe, name);
        cairo_surface_t *s = cairo_image_surface_create_from_png(path);
        if (cairo_surface_status(s) == CAIRO_STATUS_SUCCESS)
            return s;
        cairo_surface_destroy(s);
    }
    snprintf(path, sizeof(path), "assets/hamsters/%s", name);
    cairo_surface_t *s = cairo_image_surface_create_from_png(path);
    if (cairo_surface_status(s) != CAIRO_STATUS_SUCCESS) {
        fprintf(stderr, "hamsters: can't load %s (looked next to the binary and in ./assets/hamsters)\n", name);
        exit(1);
    }
    return s;
}

/* Load a sprite and scale it once to its on-screen width */
static cairo_surface_t *load_scaled(const char *name, double width)
{
    cairo_surface_t *src = load_asset(name);
    double sw = cairo_image_surface_get_width(src), sh = cairo_image_surface_get_height(src);
    double k = width / sw;
    int w = (int)ceil(sw * k), h = (int)ceil(sh * k);
    cairo_surface_t *out = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w, h);
    cairo_t *c = cairo_create(out);
    cairo_scale(c, k, k);
    cairo_set_source_surface(c, src, 0, 0);
    cairo_pattern_set_filter(cairo_get_source(c), CAIRO_FILTER_BEST);
    cairo_paint(c);
    cairo_destroy(c);
    cairo_surface_destroy(src);
    return out;
}

/* A red-faced copy: flushed all over, most of all at the head (the sprites face right) */
static cairo_surface_t *make_hot(cairo_surface_t *src)
{
    int w = cairo_image_surface_get_width(src), h = cairo_image_surface_get_height(src);
    cairo_surface_t *out = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w, h);
    cairo_t *c = cairo_create(out);
    cairo_set_source_surface(c, src, 0, 0);
    cairo_paint(c);
    cairo_set_operator(c, CAIRO_OPERATOR_ATOP);
    cairo_pattern_t *g = cairo_pattern_create_linear(0, 0, w, 0);
    cairo_pattern_add_color_stop_rgba(g, 0, 0.95, 0.20, 0.10, 0.08);
    cairo_pattern_add_color_stop_rgba(g, 0.45, 0.95, 0.18, 0.10, 0.22);
    cairo_pattern_add_color_stop_rgba(g, 1, 1.0, 0.12, 0.12, 0.55);
    cairo_set_source(c, g);
    cairo_paint(c);
    cairo_pattern_destroy(g);
    cairo_destroy(c);
    return out;
}

static cairo_surface_t *text_sprite(const char *s, double size, rgb col)
{
    cairo_surface_t *out = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, (int)(size * 1.4), (int)(size * 1.4));
    cairo_t *c = cairo_create(out);
    cairo_text_extents_t ext;
    cairo_select_font_face(c, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(c, size);
    cairo_text_extents(c, s, &ext);
    cairo_move_to(c, size * 0.7 - ext.width / 2 - ext.x_bearing, size * 0.7 - ext.height / 2 - ext.y_bearing);
    cairo_text_path(c, s);
    cairo_set_line_width(c, size * 0.18);
    cairo_set_line_join(c, CAIRO_LINE_JOIN_ROUND);
    cairo_set_source_rgba(c, 0.12, 0.06, 0.03, 0.7);
    cairo_stroke_preserve(c);
    set_rgb(c, col);
    cairo_fill(c);
    cairo_destroy(c);
    return out;
}

/* One sunflower seed, drawn pointing right, centred at 0,0 */
static void seed_path(cairo_t *c, double len)
{
    cairo_save(c);
    cairo_scale(c, len / 2, len / 4.4);
    cairo_move_to(c, -1, 0);
    cairo_curve_to(c, -0.6, -1.05, 0.5, -1.0, 1, 0);
    cairo_curve_to(c, 0.5, 1.0, -0.6, 1.05, -1, 0);
    cairo_close_path(c);
    cairo_restore(c);
}

static void draw_seed(cairo_t *c, double x, double y, double len, double rot)
{
    cairo_save(c);
    cairo_translate(c, x, y);
    cairo_rotate(c, rot);
    seed_path(c, len);
    cairo_set_source_rgb(c, 0.16, 0.13, 0.12);
    cairo_fill_preserve(c);
    cairo_set_source_rgba(c, 0.05, 0.03, 0.02, 0.9);
    cairo_set_line_width(c, 0.8);
    cairo_stroke(c);
    cairo_set_source_rgba(c, 0.92, 0.88, 0.78, 0.9);         /* the pale stripes */
    cairo_set_line_width(c, len * 0.07);
    for (int k = -1; k <= 1; k += 2) {
        cairo_move_to(c, -len * 0.36, k * len * 0.07);
        cairo_line_to(c, len * 0.36, k * len * 0.07);
    }
    cairo_stroke(c);
    cairo_restore(c);
}

/* A pile of seeds, heaped in the middle: what shows in the bowl */
static cairo_surface_t *make_seed_heap(void)
{
    int w = (int)(BOWL_RX * 2 + 8), h = 60;
    cairo_surface_t *out = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w, h);
    cairo_t *c = cairo_create(out);
    srand(7);
    for (int i = 0; i < 520; i++) {
        double x = 4 + frand() * (w - 8);
        double dx = (x - w / 2.0) / (w / 2.0);
        double top = 12 + 14 * dx * dx;                  /* dome */
        double y = top + pow(frand(), 0.7) * (h - top);
        draw_seed(c, x, y, 9 + frand() * 3, (frand() - 0.5) * 2.6);
    }
    cairo_destroy(c);
    srand((unsigned)time(NULL));
    return out;
}

static void gauge_track(cairo_t *c);
static cairo_surface_t *make_glow(double radius, rgb c, double inner);

/* Static layer: burrow painting, shelves, wheel stands and back plates, burrow doors */
static void build_bg(void)
{
    const double cx = SIZE / 2.0;
    bg_cache = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cairo_t *c = cairo_create(bg_cache);
    cairo_surface_t *img = load_asset("burrow.png");
    double k = SIZE / (double)cairo_image_surface_get_width(img);
    cairo_save(c);
    cairo_scale(c, k, k);
    cairo_set_source_surface(c, img, 0, 0);
    cairo_paint(c);
    cairo_restore(c);
    cairo_surface_destroy(img);

    /* Shelves: planks built into the walls, one under each row of wheels */
    for (int r = 0; r < 2; r++) {
        double y = CW_ROW_Y[r] + SHELF_DY, half = sqrt(240.0 * 240.0 - pow(y - cx, 2));
        cairo_pattern_t *g = cairo_pattern_create_linear(0, y, 0, y + 9);
        cairo_pattern_add_color_stop_rgb(g, 0, 0.62, 0.42, 0.24);
        cairo_pattern_add_color_stop_rgb(g, 0.3, 0.50, 0.32, 0.17);
        cairo_pattern_add_color_stop_rgb(g, 1, 0.30, 0.18, 0.09);
        cairo_rectangle(c, cx - half, y, half * 2, 9);
        cairo_set_source(c, g);
        cairo_fill(c);
        cairo_pattern_destroy(g);
        cairo_rectangle(c, cx - half, y + 9, half * 2, 6);       /* shadow below */
        cairo_set_source_rgba(c, 0, 0, 0, 0.35);
        cairo_fill(c);
        /* planks joints */
        cairo_set_source_rgba(c, 0.18, 0.10, 0.05, 0.8);
        cairo_set_line_width(c, 1);
        for (double x = cx - half + 70 + r * 30; x < cx + half; x += 120) {
            cairo_move_to(c, x + 0.5, y + 1);
            cairo_line_to(c, x + 0.5, y + 9);
        }
        cairo_stroke(c);
    }

    /* CPU wheel stands and mesh back plates */
    for (int r = 0; r < 2; r++)
        for (int i = 0; i < 8; i++) {
            double x = cx + (i - 3.5) * CW_PITCH, y = CW_ROW_Y[r];
            cairo_arc(c, x, y, CW_R - 1, 0, 2 * M_PI);
            cairo_set_source_rgba(c, 0.05, 0.03, 0.02, 0.45);
            cairo_fill(c);
            cairo_set_line_width(c, 2.2);
            cairo_set_line_cap(c, CAIRO_LINE_CAP_ROUND);
            cairo_set_source_rgba(c, 0.30, 0.28, 0.26, 1);
            cairo_move_to(c, x - 9, y + SHELF_DY + 1);
            cairo_line_to(c, x, y);
            cairo_line_to(c, x + 9, y + SHELF_DY + 1);
            cairo_stroke(c);
        }

    /* GPU wheel stands */
    for (int i = 0; i < N_GPUS; i++) {
        double x = GW_X[i];
        cairo_arc(c, x, GW_Y, GW_R - 2, 0, 2 * M_PI);
        cairo_set_source_rgba(c, 0.05, 0.03, 0.02, 0.45);
        cairo_fill(c);
        cairo_set_line_cap(c, CAIRO_LINE_CAP_ROUND);
        cairo_set_line_width(c, 6);
        cairo_set_source_rgb(c, 0.22, 0.20, 0.19);
        cairo_move_to(c, x - 34, FLOOR_Y + 2);
        cairo_line_to(c, x, GW_Y);
        cairo_line_to(c, x + 34, FLOOR_Y + 2);
        cairo_stroke(c);
        cairo_set_line_width(c, 2);
        cairo_set_source_rgba(c, 0.55, 0.52, 0.48, 0.6);
        cairo_move_to(c, x - 33, FLOOR_Y);
        cairo_line_to(c, x - 1, GW_Y + 2);
        cairo_stroke(c);
        /* feet */
        for (int s = -1; s <= 1; s += 2) {
            cairo_rectangle(c, x + s * 34 - 8, FLOOR_Y - 1, 16, 5);
            cairo_set_source_rgb(c, 0.18, 0.16, 0.15);
            cairo_fill(c);
        }
    }

    /* Burrow doors: mouse holes in the soil, dark and deep, with a stone rim */
    for (int i = 0; i < N_NVME; i++) {
        double x = HOLE_X[i], y0 = HOLE_FLOOR_Y, w = HOLE_W, h = HOLE_H;
        cairo_new_path(c);
        cairo_move_to(c, x - w / 2, y0);
        cairo_line_to(c, x - w / 2, y0 - h + w / 2);
        cairo_arc(c, x, y0 - h + w / 2, w / 2, M_PI, 2 * M_PI);
        cairo_line_to(c, x + w / 2, y0);
        cairo_close_path(c);
        cairo_path_t *p = cairo_copy_path(c);
        cairo_pattern_t *g = cairo_pattern_create_radial(x, y0 - 6, 2, x, y0 - 10, h);
        cairo_pattern_add_color_stop_rgb(g, 0, 0.10, 0.05, 0.02);
        cairo_pattern_add_color_stop_rgb(g, 1, 0.02, 0.01, 0.005);
        cairo_set_source(c, g);
        cairo_fill(c);
        cairo_pattern_destroy(g);
        cairo_append_path(c, p);
        cairo_set_line_width(c, 4);
        cairo_set_source_rgba(c, 0.42, 0.30, 0.20, 0.9);
        cairo_stroke(c);
        cairo_path_destroy(p);
        cairo_move_to(c, x - w / 2 - 4, y0 + 1);
        cairo_line_to(c, x + w / 2 + 4, y0 + 1);
        cairo_set_line_width(c, 2);
        cairo_set_source_rgba(c, 0.30, 0.20, 0.12, 0.9);
        cairo_stroke(c);
    }

    /* Soft vignette so the round edge falls away into the dark */
    cairo_pattern_t *v = cairo_pattern_create_radial(cx, cx, 170, cx, cx, 240);
    cairo_pattern_add_color_stop_rgba(v, 0, 0, 0, 0, 0);
    cairo_pattern_add_color_stop_rgba(v, 1, 0, 0, 0, 0.55);
    cairo_set_source(c, v);
    cairo_paint(c);
    cairo_pattern_destroy(v);
    gauge_track(c);
    cairo_destroy(c);

    /* The bowl's front: a glazed blue ceramic dish (the seeds go behind it) */
    bowl_front = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, (int)(BOWL_RX * 2 + 20), 60);
    c = cairo_create(bowl_front);
    cairo_translate(c, BOWL_RX + 10, 10);                       /* rim centre at 0,0 */
    double bh = BOWL_BOT_Y - BOWL_RIM_Y;
    cairo_new_path(c);
    cairo_move_to(c, -BOWL_RX, 0);
    cairo_curve_to(c, -BOWL_RX, bh * 0.7, -BOWL_RX * 0.8, bh, -BOWL_RX * 0.62, bh);
    cairo_line_to(c, BOWL_RX * 0.62, bh);
    cairo_curve_to(c, BOWL_RX * 0.8, bh, BOWL_RX, bh * 0.7, BOWL_RX, 0);
    cairo_save(c);                                              /* front half of the rim */
    cairo_scale(c, BOWL_RX, BOWL_RY);
    cairo_arc(c, 0, 0, 1, 0, M_PI);
    cairo_restore(c);
    cairo_close_path(c);
    cairo_pattern_t *g = cairo_pattern_create_linear(-BOWL_RX, 0, BOWL_RX, 0);
    cairo_pattern_add_color_stop_rgb(g, 0, 0.16, 0.30, 0.52);
    cairo_pattern_add_color_stop_rgb(g, 0.35, 0.36, 0.58, 0.82);
    cairo_pattern_add_color_stop_rgb(g, 0.55, 0.30, 0.50, 0.76);
    cairo_pattern_add_color_stop_rgb(g, 1, 0.12, 0.22, 0.42);
    cairo_set_source(c, g);
    cairo_fill_preserve(c);
    cairo_pattern_destroy(g);
    cairo_set_source_rgba(c, 0.05, 0.07, 0.15, 0.9);
    cairo_set_line_width(c, 1.6);
    cairo_stroke(c);
    /* rim lip and a highlight */
    cairo_save(c);
    cairo_scale(c, BOWL_RX, BOWL_RY);
    cairo_arc(c, 0, 0, 1, 0, 2 * M_PI);
    cairo_restore(c);
    cairo_set_line_width(c, 3);
    cairo_set_source_rgb(c, 0.62, 0.78, 0.95);
    cairo_stroke(c);
    cairo_move_to(c, -BOWL_RX + 8, 8);
    cairo_curve_to(c, -BOWL_RX + 8, 18, -BOWL_RX + 12, 24, -BOWL_RX + 18, 27);
    cairo_set_line_width(c, 2.5);
    cairo_set_source_rgba(c, 1, 1, 1, 0.35);
    cairo_stroke(c);
    cairo_destroy(c);

    seed_heap = make_seed_heap();
}

static void load_assets(void)
{
    char name[64];
    for (int i = 0; i < 4; i++) {
        snprintf(name, sizeof(name), "run_%d.png", i);
        run_spr[i] = load_scaled(name, 36);
        run_hot[i] = make_hot(run_spr[i]);
        baby_spr[i] = load_scaled(name, 18);
    }
    sleep_spr = load_scaled("sleep.png", 34);
    for (int g = 0; g < N_GPUS; g++) {
        for (int f = 0; f < 2; f++) {
            snprintf(name, sizeof(name), "gpu%d_%d.png", g, f);
            gpu_spr[g][f] = load_scaled(name, 104);
            gpu_hot[g][f] = make_hot(gpu_spr[g][f]);
        }
        snprintf(name, sizeof(name), "gpu%d_sleep.png", g);
        gpu_sleep_spr[g] = load_scaled(name, 100);
    }
    carry_spr = load_scaled("carry.png", 25);
    spit_spr = load_scaled("spit.png", 37);
    phone_spr = load_scaled("phone.png", PHONE_H * 170.0 / 200.0);
    keeper_spr = load_scaled("keeper.png", 64);
    z_spr[0] = text_sprite("z", 11, (rgb){ 0.85, 0.88, 1.0 });
    z_spr[1] = text_sprite("z", 14, (rgb){ 0.85, 0.88, 1.0 });
    z_spr[2] = text_sprite("Z", 22, (rgb){ 0.85, 0.88, 1.0 });
    bang_spr = text_sprite("!", 16, (rgb){ 1.0, 0.85, 0.3 });
    hud_cache = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, SIZE);
    lamp_glow = make_glow(95, (rgb){ 1.0, 0.65, 0.28 }, 0.04);
    gpu_glow[0] = make_glow(GW_R * 1.45, (rgb){ 0.30, 0.68, 1.0 }, 0.35);
    gpu_glow[1] = make_glow(GW_R * 1.45, (rgb){ 1.0, 0.55, 0.20 }, 0.35);
    build_bg();
}

/* ---------------------------------------------------------------- simulation */

static particle *spawn(ptype type, double x, double y, double vx, double vy, double life)
{
    for (int i = 0; i < MAX_PARTS; i++)
        if (!parts[i].alive) {
            parts[i] = (particle){ x, y, vx, vy, 0, life, frand() * 6, (frand() - 0.5) * 12, 1, type, 1 };
            return &parts[i];
        }
    return NULL;
}

/* log scale: 0 at `lo` bytes/s, 1 at `hi` */
static double log_level(double v, double lo, double hi)
{
    return v <= lo ? 0 : clamp01(log10(v / lo) / log10(hi / lo));
}

static double cpu_wheel_x(int core) { return SIZE / 2.0 + ((core % 8) - 3.5) * CW_PITCH; }
static double cpu_wheel_y(int core) { return CW_ROW_Y[core / 8]; }

/* Ease the view toward the latest readings: nothing jumps */
static void ease_view(view_t *v, const stats *g, const sys_stats *s, double dt)
{
    double k = 1 - exp(-dt * 3.0), ks = 1 - exp(-dt * 1.2);
    for (int c = 0; c < N_CORES; c++)
        v->core[c] += (s->core_load[c] - v->core[c]) * k;
    v->cpu_load += (s->cpu_load - v->cpu_load) * k;
    v->cpu_temp += ((s->cpu_temp > 0 ? s->cpu_temp : 40) - v->cpu_temp) * ks;
    for (int i = 0; i < N_GPUS; i++) {
        v->gpu_act[i] += (gpu_activity(g, i) - v->gpu_act[i]) * k;
        v->gpu_power[i] += (g->power[i] - v->gpu_power[i]) * k;
        v->gpu_temp[i] += (g->temp[i] - v->gpu_temp[i]) * ks;
    }
    static int primed;
    if (!primed && s->ram_total > 1) {                          /* start from the first reading */
        primed = 1;
        v->ram_used = s->ram_used;
        v->cpu_temp = s->cpu_temp > 0 ? s->cpu_temp : v->cpu_temp;
    }
    double used = v->ram_used;
    v->ram_total = s->ram_total > 1 ? s->ram_total : 1;
    v->ram_used += (s->ram_used - v->ram_used) * ks;
    ram_rise = dt > 0 ? (used - v->ram_used) / dt : 0;         /* GB/s freed */
    v->ram_free_frac = clamp01(1 - v->ram_used / v->ram_total);
    for (int i = 0; i < N_NVME; i++) {
        v->io_rd[i] += (log_level(s->nvme_rd[i], 2e5, 3e9) - v->io_rd[i]) * k;
        v->io_wr[i] += (log_level(s->nvme_wr[i], 2e5, 3e9) - v->io_wr[i]) * k;
    }
    v->net_rx += (log_level(s->net_rx + s->ts_rx, 2e4, 1.2e8) - v->net_rx) * k;
    v->net_tx += (log_level(s->net_tx + s->ts_tx, 2e4, 1.2e8) - v->net_tx) * k;
    double p = 0.6 * v->cpu_load + 0.2 * v->gpu_act[0] + 0.2 * v->gpu_act[1];
    v->power += (clamp01(p) - v->power) * ks;
    v->cpu_ghz += (s->cpu_mhz / 1000 - v->cpu_ghz) * k;
    v->pkg_watts += (s->pkg_watts - v->pkg_watts) * k;
    v->pkg_measured = s->pkg_watts_measured;
    v->ctxt += (log_level(s->ctxt_s, 3e4, 3e6) - v->ctxt) * k;
    v->forks += (log_level(s->forks_s, 8, 4000) - v->forks) * k;
    v->majf += (log_level(s->pgmajfault_s, 50, 20000) - v->majf) * k;
}

static void run_runner(runner *r, double load, double dt, double wake_at, double sleep_at,
                       double x, double y, double spin, double stride)
{
    r->load = load;
    if (r->asleep) {
        if (load > wake_at) {
            r->asleep = 0;
            r->sleep_t = 0;
            particle *p = spawn(P_BANG, x + 6, y - 6, 0, -18, 0.8);  /* wakes with a start */
            if (p)
                p->size = 1;
        }
    } else {
        r->sleep_t = load < sleep_at ? r->sleep_t + dt : 0;
        if (r->sleep_t > 3.0 && !r->away && r->tumble <= 0)
            r->asleep = 1;
    }
    if (!r->asleep) {
        r->angle += (0.8 + spin * load) * dt;
        r->phase += (3.0 + stride * load) * dt;
    }
}

static void simulate(const view_t *v, double dt)
{
    /* CPU hamsters: speed from their core's load, heat from Tctl (and how hard they run) */
    double tctl = clamp01((v->cpu_temp - 58) / 34);
    for (int c = 0; c < N_CORES; c++) {
        runner *r = &cpu_run[c];
        double x = cpu_wheel_x(c), y = cpu_wheel_y(c);
        run_runner(r, v->core[c], dt, 0.12, 0.05, x, y, 7.0, 13.0);
        double heat = r->asleep ? 0 : tctl * (0.35 + 0.65 * clamp01(r->load * 1.3));
        r->heat += (heat - r->heat) * fmin(1, dt * 2);
        if (r->asleep) {
            r->z_acc += dt * (0.45 + 0.05 * (c % 3));
            if (r->z_acc >= 1) {
                r->z_acc -= 1 + frand() * 0.3;
                particle *p = spawn(P_Z, x + 8, y + 6, 6 + frand() * 4, -13, 2.2);
                if (p)
                    p->size = 0;
            }
        } else if (r->heat > 0.3) {
            r->sweat_acc += dt * (r->heat - 0.3) * 7;
            while (r->sweat_acc >= 1) {
                r->sweat_acc -= 1;
                spawn(P_SWEAT, x + 7, y + 6, -20 - frand() * 30, -40 - frand() * 30, 0.7);
            }
        }
    }

    /* GPU chonks: activity spins the wheel, power throws sparks, temperature makes them sweat */
    for (int i = 0; i < N_GPUS; i++) {
        runner *r = &gpu_run[i];
        run_runner(r, v->gpu_act[i], dt, 0.06, 0.03, GW_X[i] + 20, GW_Y + 10, 2.6, 7.0);
        double heat = r->asleep ? 0 : clamp01((v->gpu_temp[i] - 52) / 30) * (0.4 + 0.6 * r->load);
        r->heat += (heat - r->heat) * fmin(1, dt * 2);
        if (r->asleep) {
            r->z_acc += dt * 0.55;
            if (r->z_acc >= 1) {
                r->z_acc -= 1;
                particle *p = spawn(P_Z, GW_X[i] + 44, GW_Y + 2, 8, -16, 2.6);
                if (p)
                    p->size = 2;
            }
            continue;
        }
        spark_acc[i] += dt * 60 * pow(clamp01((v->gpu_power[i] - 300) / 250), 1.3);
        while (spark_acc[i] >= 1) {
            spark_acc[i] -= 1;
            /* off the rim behind the chonk's feet, flung along the wheel's travel */
            double a = M_PI * (0.62 + frand() * 0.25);
            double px = GW_X[i] + cos(a) * GW_R, py = GW_Y + sin(a) * GW_R;
            double sp = 120 + frand() * 160;
            spawn(P_SPARK, px, py, -sin(a) * sp * 0.4 - 40 - frand() * 60, -80 - frand() * 150, 0.35 + frand() * 0.4);
        }
        if (r->heat > 0.3) {
            r->sweat_acc += dt * (r->heat - 0.3) * 9;
            while (r->sweat_acc >= 1) {
                r->sweat_acc -= 1;
                spawn(P_SWEAT, GW_X[i] + 28, GW_Y + 2, -30 - frand() * 40, -50 - frand() * 40, 0.8);
            }
        }
    }

    /* Context switches: neighbours tag out and swap wheels */
    swap_acc += dt * (v->ctxt < 0.05 ? 0 : 0.2 + 2.3 * v->ctxt);
    if (swap_acc >= 1) {
        swap_acc -= 1;
        int c = rand() % N_CORES, d = c + 1;
        runner *a = &cpu_run[c], *b = &cpu_run[d % N_CORES];
        int slot = -1;
        for (int i = 0; i < MAX_SWAPS; i++)
            if (!swaps[i].alive)
                slot = i;
        if (slot >= 0 && c % 8 != 7 && !a->asleep && !b->asleep && !a->away && !b->away &&
            a->tumble <= 0 && b->tumble <= 0) {
            swaps[slot] = (swap_t){ c, d, 1, 0 };
            a->away = b->away = 1;
        }
    }
    for (int i = 0; i < MAX_SWAPS; i++) {
        swap_t *w = &swaps[i];
        if (w->alive && (w->age += dt) >= SWAP_TIME) {
            w->alive = 0;
            cpu_run[w->a].away = cpu_run[w->b].away = 0;
        }
    }

    /* Major page faults: someone loses their footing and loops the loop */
    tumble_acc += dt * (v->majf < 0.03 ? 0 : 0.15 + 1.4 * v->majf);
    if (tumble_acc >= 1) {
        tumble_acc -= 1;
        int c = rand() % N_CORES;
        runner *r = &cpu_run[c];
        if (!r->asleep && !r->away && r->tumble <= 0)
            r->tumble = TUMBLE_TIME;
    }
    for (int c = 0; c < N_CORES; c++) {
        runner *r = &cpu_run[c];
        if (r->tumble > 0 && (r->tumble -= dt) <= 0) {
            r->tumble = 0;
            for (int k = 0; k < 4; k++) {                           /* seeing stars */
                particle *p = spawn(P_STAR, cpu_wheel_x(c) + (frand() - 0.5) * 14, cpu_wheel_y(c) + 8,
                                    (frand() - 0.5) * 50, -40 - frand() * 30, 0.7);
                if (p)
                    p->size = 3 + frand() * 2;
            }
        }
    }

    /* Forks: baby hamsters scurry along the shelves */
    baby_acc += dt * (v->forks < 0.05 ? 0 : 0.25 + 2.5 * v->forks);
    if (baby_acc >= 1) {
        baby_acc -= 1;
        for (int i = 0; i < MAX_BABIES; i++)
            if (!babies[i].alive) {
                int row = rand() % 2;
                double y = CW_ROW_Y[row] + SHELF_DY, half = sqrt(240.0 * 240.0 - pow(y - 240, 2)) - 30;
                double dir = rand() % 2 ? 1 : -1;
                babies[i] = (baby_t){ row, 1, 240 + (frand() * 2 - 1) * half * 0.8, dir, 30 + frand() * 25, 0,
                                      2.2 + frand() * 1.2, frand() * 4 };
                break;
            }
    }
    for (int i = 0; i < MAX_BABIES; i++) {
        baby_t *b = &babies[i];
        if (!b->alive)
            continue;
        b->age += dt;
        b->x += b->dir * b->speed * dt;
        b->phase += dt * 16;
        double y = CW_ROW_Y[b->row] + SHELF_DY, half = sqrt(240.0 * 240.0 - pow(y - 240, 2)) - 18;
        if (b->age >= b->life || fabs(b->x - 240) > half)
            b->alive = 0;
    }

    /* Burrows: writes send carriers in, reads send spitters out */
    for (int h = 0; h < N_NVME; h++)
        for (int kind = 0; kind < 2; kind++) {
            double lvl = kind ? v->io_rd[h] : v->io_wr[h];
            double rate = lvl < 0.04 ? 0 : 0.3 + 1.2 * lvl;          /* hamsters per second */
            io_acc[h][kind] += rate * dt;
            if (io_acc[h][kind] < 1)
                continue;
            int busy = 0;                                            /* no queue at the door */
            for (int i = 0; i < MAX_CRITTERS; i++)
                busy += critters[i].alive && critters[i].hole == h && critters[i].spit == kind;
            if (busy >= 1) {
                io_acc[h][kind] = 1;
                continue;
            }
            io_acc[h][kind] -= 1;
            for (int i = 0; i < MAX_CRITTERS; i++)
                if (!critters[i].alive) {
                    critters[i] = (critter){ 0, 0, kind ? 1.3 : 1.1 + 0.4 * (1 - lvl), h, kind, 1, 0 };
                    break;
                }
        }
    for (int i = 0; i < MAX_CRITTERS; i++) {
        critter *c = &critters[i];
        if (!c->alive)
            continue;
        c->age += dt;
        if (c->spit && !c->spat && c->age > c->life * 0.45) {
            c->spat = 1;                                             /* ptoo! */
            double mx = HOLE_X[c->hole] + 10 + 16, my = HOLE_FLOOR_Y - 20;
            for (int k = 0; k < 3; k++) {
                particle *p = spawn(P_SEED, mx, my, 60 + frand() * 70, -60 - frand() * 70, 0.9);
                if (p)
                    p->size = 7;
            }
        }
        if (c->age >= c->life)
            c->alive = 0;
    }

    /* Caretaker: appears on the bowl's rim and pours in seeds while memory is being freed */
    if (ram_rise > 0.4)
        keeper_t = fmax(keeper_t, 2.5);
    keeper_t -= dt;
    keeper_on = keeper_t > 0;
    keeper_a += ((keeper_on ? 1 : 0) - keeper_a) * fmin(1, dt * 5);
    if (keeper_a > 0.8) {
        pour_acc += dt * 22;
        while (pour_acc >= 1) {
            pour_acc -= 1;
            particle *p = spawn(P_SEED, BOWL_X + 12 + frand() * 4, BOWL_RIM_Y - 40, -10 - frand() * 25, 10, 0.55);
            if (p)
                p->size = 6;
        }
    }

    /* Telephone: the signal rings pulse faster with traffic */
    net_phase[0] += dt * (0.4 + 2.2 * v->net_rx);
    net_phase[1] += dt * (0.4 + 2.2 * v->net_tx);
    phone_jit += dt * (4 + 20 * fmax(v->net_rx, v->net_tx));

    for (int i = 0; i < MAX_PARTS; i++) {
        particle *p = &parts[i];
        if (!p->alive)
            continue;
        p->age += dt;
        if (p->age >= p->life) {
            p->alive = 0;
            continue;
        }
        double grav = p->type == P_SWEAT ? 260 : p->type == P_SPARK ? 320 : p->type == P_SEED ? 300 : 0;
        p->vy += grav * dt;
        p->x += p->vx * dt;
        p->y += p->vy * dt;
        p->rot += p->vrot * dt;
        if (p->type == P_Z)
            p->vx *= exp(-dt * 0.6);
    }
}

/* ---------------------------------------------------------------- render */

static void blit(cairo_t *cr, cairo_surface_t *s, double x, double y, double a)
{
    cairo_set_source_surface(cr, s, round(x), round(y));
    if (a >= 0.999)
        cairo_paint(cr);
    else
        cairo_paint_with_alpha(cr, a);
}

static double spr_w(cairo_surface_t *s) { return cairo_image_surface_get_width(s); }
static double spr_h(cairo_surface_t *s) { return cairo_image_surface_get_height(s); }

/* Wheel spokes (behind the hamster): a faint ghost when it spins fast */
/*
 * Wheel sprites. Everything round about a wheel is drawn once: the rims (the same at
 * any angle) and the spokes and rungs at a few dozen angles each, so a frame only
 * blits. Spokes repeat every 360/n degrees, so that's all the angles needed.
 */
#define CW_SPOKES       6
#define GW_SPOKES       8
#define GW_RUNGS        18
#define SPOKE_STEPS     16
#define RUNG_STEPS      10

static cairo_surface_t *cw_spokes[SPOKE_STEPS], *cw_rim, *cw_streak, *gw_spokes[SPOKE_STEPS], *gw_tyre,
                       *gw_rungs[RUNG_STEPS];

/* A square sprite centred on a wheel whose centre sits `fx` past a whole pixel */
static cairo_surface_t *wheel_canvas(double r, double fx, cairo_t **c)
{
    int h = (int)ceil(r) + 4;
    cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, 2 * h + 1, 2 * h + 1);
    *c = cairo_create(s);
    cairo_translate(*c, h + fx, h + fx);
    cairo_set_line_cap(*c, CAIRO_LINE_CAP_ROUND);
    return s;
}

static void blit_wheel(cairo_t *cr, cairo_surface_t *s, double x, double y, double a)
{
    int h = (cairo_image_surface_get_width(s) - 1) / 2;
    cairo_set_source_surface(cr, s, floor(x) - h, floor(y) - h);
    if (a >= 0.999)
        cairo_paint(cr);
    else
        cairo_paint_with_alpha(cr, a);
}

static void spoke_path(cairo_t *c, double r, int n, double ang)
{
    for (int k = 0; k < n; k++) {
        double a = ang + k * 2 * M_PI / n;
        cairo_move_to(c, cos(a) * r * 0.12, sin(a) * r * 0.12);
        cairo_line_to(c, cos(a) * r * 0.96, sin(a) * r * 0.96);
    }
}

static void build_wheels(void)
{
    cairo_t *c;
    double fx = cpu_wheel_x(0) - floor(cpu_wheel_x(0)), gx = GW_X[0] - floor(GW_X[0]);
    for (int i = 0; i < SPOKE_STEPS; i++) {
        cw_spokes[i] = wheel_canvas(CW_R, fx, &c);
        spoke_path(c, CW_R, CW_SPOKES, i * 2 * M_PI / CW_SPOKES / SPOKE_STEPS);
        cairo_set_line_width(c, 1.3);
        cairo_set_source_rgba(c, 0.72, 0.70, 0.66, 0.6);
        cairo_stroke(c);
        cairo_destroy(c);

        gw_spokes[i] = wheel_canvas(GW_R, gx, &c);
        spoke_path(c, GW_R - 3, GW_SPOKES, i * 2 * M_PI / GW_SPOKES / SPOKE_STEPS);
        cairo_set_line_width(c, 2.4);
        cairo_set_source_rgba(c, 0.72, 0.70, 0.66, 0.6);
        cairo_stroke(c);
        cairo_arc(c, 0, 0, GW_R * 0.18, 0, 2 * M_PI);
        cairo_set_source_rgb(c, 0.35, 0.33, 0.31);
        cairo_fill(c);
        cairo_destroy(c);
    }
    cw_rim = wheel_canvas(CW_R + 2, fx, &c);
    cairo_arc(c, 0, 0, CW_R, 0, 2 * M_PI);
    cairo_set_line_width(c, 3.2);
    cairo_set_source_rgb(c, 0.28, 0.27, 0.26);
    cairo_stroke_preserve(c);
    cairo_set_line_width(c, 1.6);
    cairo_set_source_rgb(c, 0.80, 0.78, 0.74);
    cairo_stroke(c);
    cairo_arc(c, 0, 0, 2.2, 0, 2 * M_PI);
    cairo_set_source_rgb(c, 0.55, 0.53, 0.5);
    cairo_fill(c);
    cairo_destroy(c);

    cw_streak = wheel_canvas(CW_R + 8, fx, &c);
    cairo_set_line_width(c, 1.2);
    for (int k = 0; k < 2; k++) {
        cairo_new_path(c);
        cairo_arc(c, 0, 0, CW_R + 3 + k * 2.5, M_PI * 0.62, M_PI * (0.95 + 0.1 * k));
        cairo_set_source_rgba(c, 1, 0.95, 0.85, 0.45);
        cairo_stroke(c);
    }
    cairo_destroy(c);

    gw_tyre = wheel_canvas(GW_R + 5, gx, &c);
    cairo_arc(c, 0, 0, GW_R, 0, 2 * M_PI);
    cairo_set_line_width(c, 7);
    cairo_set_source_rgb(c, 0.20, 0.19, 0.19);
    cairo_stroke(c);
    cairo_destroy(c);
    for (int i = 0; i < RUNG_STEPS; i++) {
        gw_rungs[i] = wheel_canvas(GW_R + 5, gx, &c);
        cairo_set_line_width(c, 1.5);
        cairo_set_source_rgba(c, 0.12, 0.11, 0.1, 0.8);
        for (int k = 0; k < GW_RUNGS; k++) {
            double a = (k + i / (double)RUNG_STEPS) * 2 * M_PI / GW_RUNGS;
            cairo_move_to(c, cos(a) * (GW_R - 3), sin(a) * (GW_R - 3));
            cairo_line_to(c, cos(a) * (GW_R + 3), sin(a) * (GW_R + 3));
        }
        cairo_stroke(c);
        cairo_destroy(c);
    }
}

/* Sprite index for a wheel turned by `ang`, with `n` spokes (or rungs) and `steps` sprites */
static int wheel_step(double ang, int n, int steps)
{
    double period = 2 * M_PI / n;
    double u = fmod(ang, period) / period;
    if (u < 0)
        u += 1;
    return (int)(u * steps + 0.5) % steps;
}

static void draw_cpu_wheels(cairo_t *cr, const view_t *v)
{
    (void)v;
    for (int c = 0; c < N_CORES; c++) {
        const runner *r = &cpu_run[c];
        double x = cpu_wheel_x(c), y = cpu_wheel_y(c);
        blit_wheel(cr, cw_spokes[wheel_step(r->angle, CW_SPOKES, SPOKE_STEPS)], x, y, 1);
        if (!r->asleep && r->load > 0.5)                /* a ghost of the spokes when it's really going */
            blit_wheel(cr, cw_spokes[wheel_step(r->angle - 0.18, CW_SPOKES, SPOKE_STEPS)], x, y, 0.42);
    }
    /* hamsters */
    for (int c = 0; c < N_CORES; c++) {
        const runner *r = &cpu_run[c];
        double x = cpu_wheel_x(c), y = cpu_wheel_y(c);
        if (r->asleep) {
            blit(cr, sleep_spr, x - spr_w(sleep_spr) / 2, y + CW_R - 2 - spr_h(sleep_spr), 1);
            continue;
        }
        if (r->away)                        /* mid-air, swapping wheels */
            continue;
        int f = (int)r->phase % 4;
        if (r->tumble > 0) {                /* carried round the loop by its own wheel */
            double u = 1 - r->tumble / TUMBLE_TIME;
            u = u * u * (3 - 2 * u);
            cairo_save(cr);
            cairo_translate(cr, x, y);
            cairo_rotate(cr, u * 2 * M_PI);
            cairo_set_source_surface(cr, run_spr[1], -spr_w(run_spr[1]) / 2 + 1, CW_R - 2.5 - spr_h(run_spr[1]));
            cairo_paint(cr);
            if (r->heat > 0.02) {
                cairo_set_source_surface(cr, run_hot[1], -spr_w(run_spr[1]) / 2 + 1, CW_R - 2.5 - spr_h(run_spr[1]));
                cairo_paint_with_alpha(cr, fmin(1, r->heat * 1.1));
            }
            cairo_restore(cr);
            continue;
        }
        double bob = -fabs(sin(r->phase * M_PI / 2)) * 1.5 * (0.3 + r->load);
        double hx = x - spr_w(run_spr[f]) / 2 + 1, hy = y + CW_R - 2.5 - spr_h(run_spr[f]) + bob;
        blit(cr, run_spr[f], hx, hy, 1);
        if (r->heat > 0.02)
            blit(cr, run_hot[f], hx, hy, fmin(1, r->heat * 1.1));
    }
    /* rims in front, plus speed streaks when flat out */
    for (int c = 0; c < N_CORES; c++) {
        const runner *r = &cpu_run[c];
        double x = cpu_wheel_x(c), y = cpu_wheel_y(c);
        blit_wheel(cr, cw_rim, x, y, 1);
        if (!r->asleep && r->load > 0.55)
            blit_wheel(cr, cw_streak, x, y, (r->load - 0.55) / 0.45);
    }
}

/* Hamsters hopping between wheels, and babies running along the shelves */
static void draw_hoppers(cairo_t *cr)
{
    for (int i = 0; i < MAX_SWAPS; i++) {
        const swap_t *w = &swaps[i];
        if (!w->alive)
            continue;
        double u = w->age / SWAP_TIME;
        for (int k = 0; k < 2; k++) {
            int from = k ? w->b : w->a, to = k ? w->a : w->b;
            double x0 = cpu_wheel_x(from), x1 = cpu_wheel_x(to), y = cpu_wheel_y(from) + CW_R - 2.5;
            double x = x0 + (x1 - x0) * u, hop = sin(u * M_PI) * (k ? 20 : 30);
            cairo_surface_t *sp = run_spr[k ? 3 : 1];
            cairo_save(cr);
            cairo_translate(cr, x, y - hop);
            if (x1 < x0)
                cairo_scale(cr, -1, 1);                 /* face the way it's jumping */
            cairo_rotate(cr, (u - 0.5) * -0.6);
            cairo_set_source_surface(cr, sp, -spr_w(sp) / 2, -spr_h(sp));
            cairo_paint(cr);
            double heat = cpu_run[from].heat;
            if (heat > 0.02) {
                cairo_set_source_surface(cr, run_hot[k ? 3 : 1], -spr_w(sp) / 2, -spr_h(sp));
                cairo_paint_with_alpha(cr, fmin(1, heat * 1.1));
            }
            cairo_restore(cr);
        }
    }
    for (int i = 0; i < MAX_BABIES; i++) {
        const baby_t *b = &babies[i];
        if (!b->alive)
            continue;
        double a = fmin(1, fmin(b->age * 4, (b->life - b->age) * 4));
        cairo_surface_t *sp = baby_spr[(int)b->phase % 4];
        cairo_save(cr);
        cairo_translate(cr, b->x, CW_ROW_Y[b->row] + SHELF_DY + 1 - fabs(sin(b->phase * M_PI / 2)) * 1.2);
        if (b->dir < 0)
            cairo_scale(cr, -1, 1);
        cairo_set_source_surface(cr, sp, -spr_w(sp) / 2, -spr_h(sp));
        cairo_paint_with_alpha(cr, clamp01(a));
        cairo_restore(cr);
    }
}

static void draw_gpu_wheels(cairo_t *cr, const view_t *v)
{
    static const rgb col[N_GPUS] = { { 0.30, 0.68, 1.0 }, { 1.0, 0.55, 0.20 } };
    for (int i = 0; i < N_GPUS; i++) {
        const runner *r = &gpu_run[i];
        double x = GW_X[i], y = GW_Y, act = r->asleep ? 0 : r->load;

        /* coloured glow behind the wheel, stronger with power */
        double glow = 0.12 + 0.5 * clamp01((v->gpu_power[i] - 40) / 500);
        blit(cr, gpu_glow[i], x - spr_w(gpu_glow[i]) / 2, y - spr_h(gpu_glow[i]) / 2, glow);

        blit_wheel(cr, gw_spokes[wheel_step(r->angle, GW_SPOKES, SPOKE_STEPS)], x, y, 1);
        if (act > 0.5)
            blit_wheel(cr, gw_spokes[wheel_step(r->angle - 0.12, GW_SPOKES, SPOKE_STEPS)], x, y, 0.42);

        /* the chonk */
        if (r->asleep) {
            cairo_surface_t *s = gpu_sleep_spr[i];
            blit(cr, s, x - spr_w(s) / 2 + 4, y + GW_R - 5 - spr_h(s), 1);
        } else {
            int f = (int)r->phase % 2;
            cairo_surface_t *s = gpu_spr[i][f];
            double bob = -fabs(sin(r->phase * M_PI)) * 3 * (0.3 + act);
            double hx = x - spr_w(s) / 2 + 2, hy = y + GW_R - 6 - spr_h(s) + bob;
            blit(cr, s, hx, hy, 1);
            if (r->heat > 0.02)
                blit(cr, gpu_hot[i][f], hx, hy, fmin(1, r->heat));
        }

        /* rim: dark tyre, metal band, rungs sweeping round, lit in the GPU's colour */
        blit_wheel(cr, gw_tyre, x, y, 1);
        cairo_new_path(cr);
        cairo_arc(cr, x, y, GW_R, 0, 2 * M_PI);
        cairo_set_line_width(cr, 3);
        rgb band = lerp((rgb){ 0.78, 0.76, 0.72 }, col[i], 0.25 + 0.6 * act);
        set_rgb(cr, band);
        cairo_stroke(cr);
        blit_wheel(cr, gw_rungs[wheel_step(r->angle, GW_RUNGS, RUNG_STEPS)], x, y, 1);
        if (act > 0.5) {                                             /* whoosh */
            double a = (act - 0.5) / 0.5;
            cairo_set_line_width(cr, 2);
            for (int k = 0; k < 3; k++) {
                cairo_new_path(cr);
                cairo_arc(cr, x, y, GW_R + 7 + k * 4, M_PI * (0.55 + 0.05 * k), M_PI * (1.0 + 0.04 * k));
                cairo_set_source_rgba(cr, col[i].r, col[i].g, col[i].b, (0.55 - k * 0.15) * a);
                cairo_stroke(cr);
            }
        }
    }
}

static void draw_phone(cairo_t *cr, const view_t *v)
{
    double w = spr_w(phone_spr), h = spr_h(phone_spr);
    double jit = fmax(v->net_rx, v->net_tx) > 0.5 ? sin(phone_jit) * 1.2 : 0;
    double x0 = PHONE_X - w / 2, y0 = PHONE_BOT_Y - h + jit;
    /* signal rings: incoming (blue) close in on the handset, outgoing (orange) leave the mouth */
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    for (int dir = 0; dir < 2; dir++) {
        double lvl = dir ? v->net_tx : v->net_rx;
        if (lvl < 0.02)
            continue;
        double ox = dir ? x0 + w * 0.70 : x0 + w * 0.28, oy = y0 + h * (dir ? 0.40 : 0.30);
        double base = dir ? -0.2 : M_PI + 0.2;
        rgb c = dir ? (rgb){ 1.0, 0.62, 0.25 } : (rgb){ 0.45, 0.78, 1.0 };
        cairo_set_line_width(cr, 2.2);
        for (int k = 0; k < 3; k++) {
            double f = fmod(net_phase[dir] + k / 3.0, 1.0);
            if (!dir)
                f = 1 - f;                                           /* incoming: rings shrink */
            double rr = 7 + f * 16;
            double a = (0.35 + 0.65 * lvl) * sin(f * M_PI);
            cairo_new_path(cr);
            cairo_arc(cr, ox, oy, rr, base - 0.6, base + 0.6);
            cairo_set_source_rgba(cr, c.r, c.g, c.b, a);
            cairo_stroke(cr);
        }
    }
    blit(cr, phone_spr, x0, y0, 1);
}

static void draw_bowl(cairo_t *cr, const view_t *v)
{
    double fill = v->ram_free_frac;
    /* interior shadow, then the heap sunk as deep as the bowl is empty */
    cairo_save(cr);
    cairo_translate(cr, BOWL_X, BOWL_RIM_Y);
    cairo_scale(cr, BOWL_RX - 1, BOWL_RY - 1);
    cairo_arc(cr, 0, 0, 1, 0, 2 * M_PI);
    cairo_restore(cr);
    cairo_set_source_rgb(cr, 0.10, 0.12, 0.2);
    cairo_fill(cr);
    if (fill > 0.01) {
        double hw = spr_w(seed_heap);
        double top = BOWL_RIM_Y - 36 + (1 - fill) * 50;          /* full: heaped over the rim */
        cairo_save(cr);
        /* seeds show above the rim only inside the bowl's width, and inside the mouth */
        cairo_rectangle(cr, BOWL_X - BOWL_RX + 2, BOWL_RIM_Y - 40, BOWL_RX * 2 - 4, 40);
        cairo_save(cr);
        cairo_translate(cr, BOWL_X, BOWL_RIM_Y);
        cairo_scale(cr, BOWL_RX - 1, BOWL_RY - 1);
        cairo_arc(cr, 0, 0, 1, 0, 2 * M_PI);
        cairo_restore(cr);
        cairo_clip(cr);
        /* the heap narrows as it sinks */
        double k = 0.55 + 0.45 * fill;
        cairo_translate(cr, BOWL_X, top);
        cairo_scale(cr, k, 1);
        cairo_set_source_surface(cr, seed_heap, -hw / 2, 0);
        cairo_paint(cr);
        cairo_restore(cr);
    }
    blit(cr, bowl_front, BOWL_X - BOWL_RX - 10, BOWL_RIM_Y - 10, 1);

    if (keeper_a > 0.02) {
        double w = spr_w(keeper_spr), h = spr_h(keeper_spr);
        double hop = (1 - keeper_a) * 10;
        blit(cr, keeper_spr, BOWL_X + BOWL_RX - w * 0.62, BOWL_RIM_Y + 4 - h + hop, keeper_a);
    }
}

static void draw_burrows(cairo_t *cr, const view_t *v)
{
    /* warm light inside a busy burrow */
    for (int h = 0; h < N_NVME; h++) {
        double a = fmax(v->io_rd[h], v->io_wr[h]);
        if (a < 0.02)
            continue;
        double x = HOLE_X[h], y = HOLE_FLOOR_Y - 12;
        cairo_pattern_t *g = cairo_pattern_create_radial(x, y, 1, x, y, 22);
        cairo_pattern_add_color_stop_rgba(g, 0, 1.0, 0.7, 0.3, 0.45 * a);
        cairo_pattern_add_color_stop_rgba(g, 1, 1.0, 0.5, 0.2, 0);
        cairo_set_source(cr, g);
        cairo_arc(cr, x, y, 22, 0, 2 * M_PI);
        cairo_fill(cr);
        cairo_pattern_destroy(g);
    }
    for (int i = 0; i < MAX_CRITTERS; i++) {
        const critter *c = &critters[i];
        if (!c->alive)
            continue;
        double u = c->age / c->life, hx = HOLE_X[c->hole];
        cairo_surface_t *s = c->spit ? spit_spr : carry_spr;
        double w = spr_w(s), h = spr_h(s), x, sc, a;
        if (!c->spit) {                     /* walk in from the left and vanish into the dark */
            x = hx - 28 + 28 * u;
            sc = u < 0.75 ? 1 : 1 - (u - 0.75) / 0.25 * 0.4;
            a = fmin(1, u * 6) * (u < 0.75 ? 1 : 1 - (u - 0.75) / 0.25);
        } else {                            /* pop out, spit, duck back */
            double out = u < 0.25 ? u / 0.25 : u > 0.8 ? 1 - (u - 0.8) / 0.2 : 1;
            x = hx + 10 * out;
            sc = 0.6 + 0.4 * out;
            a = out;
        }
        double bob = c->spit ? 0 : -fabs(sin(c->age * 12)) * 1.5;
        cairo_save(cr);
        cairo_translate(cr, x, HOLE_FLOOR_Y + bob);
        cairo_scale(cr, sc, sc);
        cairo_set_source_surface(cr, s, -w / 2, -h);
        cairo_paint_with_alpha(cr, a);
        cairo_restore(cr);
    }
}

static void draw_particles(cairo_t *cr)
{
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    for (int i = 0; i < MAX_PARTS; i++) {
        const particle *p = &parts[i];
        if (!p->alive)
            continue;
        double u = p->age / p->life;
        switch (p->type) {
        case P_SWEAT:
            cairo_save(cr);
            cairo_translate(cr, p->x, p->y);
            cairo_rotate(cr, atan2(p->vy, p->vx) - M_PI / 2);
            cairo_move_to(cr, 0, 3.2);
            cairo_curve_to(cr, 2.2, 0.5, 1.8, -1.8, 0, -1.8);
            cairo_curve_to(cr, -1.8, -1.8, -2.2, 0.5, 0, 3.2);
            cairo_set_source_rgba(cr, 0.65, 0.88, 1.0, 0.9 * (1 - u));
            cairo_fill(cr);
            cairo_restore(cr);
            break;
        case P_SPARK: {
            double a = 1 - u;
            cairo_move_to(cr, p->x, p->y);
            cairo_line_to(cr, p->x - p->vx * 0.035, p->y - p->vy * 0.035);
            cairo_set_line_width(cr, 1.8);
            cairo_set_source_rgba(cr, 1.0, 0.85 - 0.4 * u, 0.4 - 0.3 * u, a);
            cairo_stroke(cr);
            break;
        }
        case P_SEED:
            draw_seed(cr, p->x, p->y, p->size, p->rot);
            break;
        case P_Z: {
            cairo_surface_t *s = z_spr[(int)p->size + (p->size < 2 && u > 0.5)];
            double a = u < 0.15 ? u / 0.15 : 1 - (u - 0.15) / 0.85;
            blit(cr, s, p->x - spr_w(s) / 2, p->y - spr_h(s) / 2, a * 0.9);
            break;
        }
        case P_STAR: {
            double r = p->size * (1 - 0.4 * u);
            cairo_save(cr);
            cairo_translate(cr, p->x, p->y);
            cairo_rotate(cr, p->rot);
            cairo_move_to(cr, 0, -r);
            for (int k = 1; k < 8; k++) {
                double rr = k % 2 ? r * 0.38 : r;
                cairo_line_to(cr, sin(k * M_PI / 4) * rr, -cos(k * M_PI / 4) * rr);
            }
            cairo_close_path(cr);
            cairo_set_source_rgba(cr, 1.0, 0.9, 0.35, 1 - u * u);
            cairo_fill(cr);
            cairo_restore(cr);
            break;
        }
        case P_BANG:
            blit(cr, bang_spr, p->x - spr_w(bang_spr) / 2, p->y - spr_h(bang_spr) / 2, 1 - u * u);
            break;
        }
    }
}

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
    cairo_set_source_rgba(cr, 0.08, 0.04, 0.02, 0.85);
    cairo_stroke_preserve(cr);
    set_rgb(cr, c);
    cairo_fill(cr);
}

/* Rim gauges: wheel "RPM" (CPU clock) on the left, hamster-power (package watts) on the right */
#define GAUGE_R     229.0
#define GAUGE_A0    (160 * M_PI / 180)      /* left gauge, bottom end */
#define GAUGE_A1    (222 * M_PI / 180)      /* left gauge, top end; the right one is mirrored */
#define GHZ_MIN     0.4
#define GHZ_MAX     5.8
#define WATTS_MAX   230.0

static double gauge_ghz(const view_t *v) { return clamp01((v->cpu_ghz - GHZ_MIN) / (GHZ_MAX - GHZ_MIN)); }
static double gauge_watts(const view_t *v) { return clamp01(v->pkg_watts / WATTS_MAX); }
static const rgb GHZ_COL = { 1.0, 0.80, 0.32 };
static rgb watts_col(double f) { return lerp((rgb){ 1.0, 0.86, 0.35 }, (rgb){ 1.0, 0.32, 0.18 }, clamp01(f)); }

static void gauge_track(cairo_t *c)
{
    const double cx = SIZE / 2.0;
    cairo_set_line_cap(c, CAIRO_LINE_CAP_ROUND);
    for (int side = 0; side < 2; side++) {
        double a0 = side ? M_PI - GAUGE_A1 : GAUGE_A0, a1 = side ? M_PI - GAUGE_A0 : GAUGE_A1;
        cairo_new_path(c);
        cairo_arc(c, cx, cx, GAUGE_R, a0, a1);
        cairo_set_line_width(c, 11);
        cairo_set_source_rgba(c, 0.03, 0.015, 0.01, 0.75);
        cairo_stroke_preserve(c);
        cairo_set_line_width(c, 7);
        cairo_set_source_rgba(c, 0.30, 0.21, 0.14, 1);
        cairo_stroke(c);
        cairo_set_line_width(c, 1.2);
        cairo_set_source_rgba(c, 0.75, 0.62, 0.48, 0.8);
        for (int k = 0; k <= 8; k++) {
            double a = a0 + (a1 - a0) * k / 8.0;
            cairo_move_to(c, cx + cos(a) * (GAUGE_R - 9), cx + sin(a) * (GAUGE_R - 9));
            cairo_line_to(c, cx + cos(a) * (GAUGE_R - 6), cx + sin(a) * (GAUGE_R - 6));
        }
        cairo_stroke(c);
    }
}

static void draw_gauges(cairo_t *cr, const view_t *v)
{
    const double cx = SIZE / 2.0;
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    for (int side = 0; side < 2; side++) {
        double f = side ? gauge_watts(v) : gauge_ghz(v);
        rgb col = side ? watts_col(f) : GHZ_COL;
        double span = (GAUGE_A1 - GAUGE_A0) * fmax(f, 0.02), end;
        cairo_new_path(cr);
        if (side) {
            end = M_PI - GAUGE_A0 - span;
            cairo_arc_negative(cr, cx, cx, GAUGE_R, M_PI - GAUGE_A0, end);
        } else {
            end = GAUGE_A0 + span;
            cairo_arc(cr, cx, cx, GAUGE_R, GAUGE_A0, end);
        }
        cairo_set_line_width(cr, 5);
        set_rgb(cr, col);
        cairo_stroke(cr);
        /* bright tip */
        double tx = cx + cos(end) * GAUGE_R, ty = cx + sin(end) * GAUGE_R;
        cairo_arc(cr, tx, ty, 4.5, 0, 2 * M_PI);
        cairo_set_source_rgba(cr, 1, 0.97, 0.85, 0.95);
        cairo_fill(cr);
    }
}

static cairo_surface_t *make_glow(double radius, rgb c, double inner)
{
    int d = (int)ceil(radius * 2);
    cairo_surface_t *out = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, d, d);
    cairo_t *g = cairo_create(out);
    cairo_pattern_t *p = cairo_pattern_create_radial(d / 2.0, d / 2.0, radius * inner, d / 2.0, d / 2.0, radius);
    cairo_pattern_add_color_stop_rgba(p, 0, c.r, c.g, c.b, 0.55);
    cairo_pattern_add_color_stop_rgba(p, 0.4, c.r, c.g, c.b, 0.22);
    cairo_pattern_add_color_stop_rgba(p, 1, c.r, c.g, c.b, 0);
    cairo_set_source(g, p);
    cairo_paint(g);
    cairo_pattern_destroy(p);
    cairo_destroy(g);
    return out;
}

/* Lightning bolt for the hamster-power reading */
static void bolt(cairo_t *cr, double x, double y, double h, rgb c)
{
    cairo_save(cr);
    cairo_translate(cr, x, y);
    cairo_scale(cr, h / 10, h / 10);
    cairo_move_to(cr, 1.5, -5);
    cairo_line_to(cr, -2.8, 0.8);
    cairo_line_to(cr, -0.2, 0.8);
    cairo_line_to(cr, -1.5, 5);
    cairo_line_to(cr, 2.8, -0.9);
    cairo_line_to(cr, 0.2, -0.9);
    cairo_close_path(cr);
    cairo_restore(cr);
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
    cairo_set_line_width(cr, 3);
    cairo_set_source_rgba(cr, 0.08, 0.04, 0.02, 0.85);
    cairo_stroke_preserve(cr);
    set_rgb(cr, c);
    cairo_fill(cr);
}

static double text_w(cairo_t *cr, double size, const char *s)
{
    cairo_text_extents_t ext;
    cairo_select_font_face(cr, "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, size);
    cairo_text_extents(cr, s, &ext);
    return ext.x_advance;
}

/*
 * Numbers: clock, CPU temperature and package watts up top (each next to its gauge),
 * watts under each GPU wheel, RAM used on the bowl. Redrawn at most 4 times a second,
 * and only when a shown number changes.
 */
static void update_hud(const view_t *v, double t)
{
    static double next, ghz, temp, watts[N_GPUS], ram, pkg;
    char key[160], txt[64];
    if (t >= next || t < next - 1) {
        next = t + 0.25;
        /* only move when past half a step beyond the shown value: no flicker */
        if (fabs(v->cpu_ghz - ghz) > 0.07)
            ghz = round(v->cpu_ghz * 10) / 10;
        if (fabs(v->cpu_temp - temp) > 0.7)
            temp = round(v->cpu_temp);
        if (fabs(v->pkg_watts - pkg) > 1.5)
            pkg = round(v->pkg_watts);
        for (int i = 0; i < N_GPUS; i++)
            if (fabs(v->gpu_power[i] - watts[i]) > 1.5)
                watts[i] = round(v->gpu_power[i]);
        if (fabs(v->ram_used - ram) > 0.7)
            ram = round(v->ram_used);
    }
    snprintf(key, sizeof(key), "%.1f|%.0f|%.0f|%.0f|%.0f|%.0f|%d", ghz, temp, pkg, watts[0], watts[1], ram,
             v->pkg_measured);
    if (!strcmp(key, hud_key))
        return;
    strcpy(hud_key, key);
    cairo_t *cr = cairo_create(hud_cache);
    cairo_set_operator(cr, CAIRO_OPERATOR_CLEAR);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    const double ty = 76, ts = 23;
    snprintf(txt, sizeof(txt), "%.1f GHz", ghz);
    soft_text(cr, 138, ty, ts, GHZ_COL, txt);
    snprintf(txt, sizeof(txt), "%.0f°C", temp);
    soft_text(cr, 240, ty, ts, lerp((rgb){ 1.0, 0.94, 0.82 }, (rgb){ 1.0, 0.35, 0.25 }, clamp01((temp - 60) / 30)), txt);
    snprintf(txt, sizeof(txt), "%s%.0f W", v->pkg_measured ? "" : "~", pkg);
    rgb wc = watts_col(pkg / WATTS_MAX);
    double w = text_w(cr, ts, txt), bx = 350 - (w + 14) / 2;
    bolt(cr, bx + 5, ty, 19, wc);
    soft_text(cr, bx + 14 + w / 2, ty, ts, wc, txt);
    static const rgb gcol[N_GPUS] = { { 0.55, 0.80, 1.0 }, { 1.0, 0.70, 0.40 } };
    for (int i = 0; i < N_GPUS; i++) {
        snprintf(txt, sizeof(txt), "%.0f W", watts[i]);
        soft_text(cr, GW_X[i] + (i ? 6 : -6), 384, 23, gcol[i], txt);
    }
    snprintf(txt, sizeof(txt), "%.0f GB", ram);
    soft_text(cr, BOWL_X, BOWL_RIM_Y + 18, 22, (rgb){ 1.0, 0.97, 0.90 }, txt);
    cairo_destroy(cr);
}

static void render(cairo_t *cr, const view_t *v, double t)
{
    cairo_set_source_surface(cr, bg_cache, 0, 0);
    cairo_paint(cr);

    /* The lamps run on hamster power: brighter the harder they work */
    double lamp = 0.15 + 0.55 * v->power;
    blit(cr, lamp_glow, 242 - spr_w(lamp_glow) / 2, 42 - spr_h(lamp_glow) / 2, lamp);
    draw_gauges(cr, v);

    draw_cpu_wheels(cr, v);
    draw_hoppers(cr);
    draw_gpu_wheels(cr, v);
    draw_phone(cr, v);
    draw_burrows(cr, v);
    update_hud(v, t);
    draw_bowl(cr, v);
    draw_particles(cr);
    cairo_set_source_surface(cr, hud_cache, 0, 0);
    cairo_paint(cr);
}

/* ---------------------------------------------------------------- main */

static int plant_idle(const view_t *v)
{
    int awake = 0;
    for (int c = 0; c < N_CORES; c++)
        awake += !cpu_run[c].asleep;
    for (int i = 0; i < N_GPUS; i++)
        awake += !gpu_run[i].asleep;
    double io = 0;
    for (int h = 0; h < N_NVME; h++)
        io = fmax(io, fmax(v->io_rd[h], v->io_wr[h]));
    return awake <= 2 && io < 0.04 && fmax(v->net_rx, v->net_tx) < 0.1 && !keeper_on;
}

int main(int argc, char **argv)
{
    cairo_surface_t *surf;
    cairo_t *cr;
    tjhandle tj;
    unsigned char *jpeg = NULL;
    stats g = { 0 };
    sys_stats sys = { 0 };
    view_t v = { 0 };
    double last, next_poll = 0, t0;
    int fd = -1, bench = 0, showcase = 0, sensors = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--demo"))
            demo = 1;
        else if (!strcmp(argv[i], "--showcase"))
            showcase = demo = 1;
        else if (!strcmp(argv[i], "--bench"))
            bench = 1;
        else if (!strcmp(argv[i], "--sensors"))
            sensors = 1;
        /* --gpu-load: accepted, this display always runs on GPU activity */
    }

    if (sensors) {                      /* print what the sensors read, for checking a new machine */
        gpus_init();
        sys_init();
        for (int n = 0; n < 4; n++) {
            sys_poll(&sys, now_s());
            gpus_poll(&g);
            if (n)
                printf("cpu %.0f%% %.1f GHz (max %.1f) %.0f C  pkg %.0f W%s  ctxt %.0f/s intr %.0f/s forks %.0f/s  "
                       "majflt %.0f/s  psi cpu/io/mem %.1f/%.1f/%.1f  ram %.1f/%.1f GB  nvme r/w %.1f/%.1f MB/s  "
                       "net rx/tx %.0f/%.0f kB/s  gpu %.0f/%.0f W %d/%d C\n",
                       sys.cpu_load * 100, sys.cpu_mhz / 1000, sys.cpu_mhz_max / 1000, sys.cpu_temp, sys.pkg_watts,
                       sys.pkg_watts_measured ? "" : " (estimated)", sys.ctxt_s, sys.intr_s, sys.forks_s,
                       sys.pgmajfault_s, sys.psi_cpu, sys.psi_io, sys.psi_mem, sys.ram_used, sys.ram_total,
                       (sys.nvme_rd[0] + sys.nvme_rd[1] + sys.nvme_rd[2]) / 1e6,
                       (sys.nvme_wr[0] + sys.nvme_wr[1] + sys.nvme_wr[2]) / 1e6, sys.net_rx / 1e3, sys.net_tx / 1e3,
                       g.power[0], g.power[1], g.temp[0], g.temp[1]);
            sleep(1);
        }
        return 0;
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);
    srand((unsigned)time(NULL));
    load_assets();
    build_wheels();
    for (int c = 0; c < N_CORES; c++)
        cpu_run[c].asleep = 1, cpu_run[c].z_acc = frand(), cpu_run[c].phase = frand() * 4;
    for (int i = 0; i < N_GPUS; i++)
        gpu_run[i].asleep = 1, gpu_run[i].z_acc = i * 0.5;
    v.cpu_temp = 45;
    v.ram_used = 20;

    surf = cairo_image_surface_create(CAIRO_FORMAT_RGB24, SIZE, SIZE);
    cr = cairo_create(surf);
    tj = tj3Init(TJINIT_COMPRESS);
    tj3Set(tj, TJPARAM_QUALITY, JPEG_QUALITY);
    tj3Set(tj, TJPARAM_SUBSAMP, TJSAMP_420);

    if (bench) {
        /* showcase moments: flat out, a build in progress, everyone asleep */
        struct { double t; const char *png; } scenes[] = {
            { 21, "hamsters_preview.png" },
            { 12, "hamsters_mid.png" },
            { 2, "hamsters_idle.png" },
        };
        for (int k = 0; k < 3; k++) {
            srand(3 + k);
            int n = 10 * FPS_BUSY;
            size_t len = 0;
            double b0 = 0;
            for (int i = 0; i < n; i++) {
                double t = scenes[k].t - 10 + i / (double)FPS_BUSY;
                if (i == n - 60)
                    b0 = now_s();
                showcase_poll(&g, &sys, t < 0 ? 0 : t);
                ease_view(&v, &g, &sys, 1.0 / FPS_BUSY);
                simulate(&v, 1.0 / FPS_BUSY);
                render(cr, &v, t);
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
            showcase_poll(&g, &sys, t - t0);
        } else if (t >= next_poll) {
            next_poll = t + POLL_S;
            if (demo) {
                /* simulate into a copy; scaling the live struct in place would compound */
                static stats demo_g;
                static sys_stats demo_s;
                demo_poll(&demo_g, &demo_s, t - t0);
                g = demo_g;
                sys = demo_s;
            } else {
                gpus_poll(&g);
                sys_poll(&sys, t);
            }
        }

        ease_view(&v, &g, &sys, dt);
        simulate(&v, dt);
        render(cr, &v, t - t0);
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

        double spare = 1.0 / (plant_idle(&v) && !showcase ? FPS_IDLE : FPS_BUSY) - (now_s() - t);
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
