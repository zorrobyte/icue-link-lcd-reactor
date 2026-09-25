CFLAGS  ?= -O2 -march=native -Wall -Wextra
CFLAGS  += $(shell pkg-config --cflags cairo libturbojpeg) -I/opt/cuda/include
LDLIBS  += $(shell pkg-config --libs cairo libturbojpeg) -lnvidia-ml -lm

DISPLAYS = reactor singularity synapse horizon brrr plasma fishbowl autumn thirst tears koi tears2

all: $(DISPLAYS)

reactor singularity synapse brrr plasma fishbowl autumn thirst tears koi tears2: %: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

horizon: horizon.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS) -lpcap -lpthread

install: all
	for d in $(DISPLAYS); do install -Dm755 $$d $(DESTDIR)/opt/llm-reactor/$$d; done
	for f in assets/*/*.png; do install -Dm644 $$f $(DESTDIR)/opt/llm-reactor/$$f; done

clean:
	rm -f $(DISPLAYS) *_preview.png
