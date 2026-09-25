CFLAGS  ?= -O2 -march=native -Wall -Wextra
CFLAGS  += $(shell pkg-config --cflags cairo libturbojpeg) -I/opt/cuda/include
LDLIBS  += $(shell pkg-config --libs cairo libturbojpeg) -lnvidia-ml -lm

reactor: reactor.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

install: reactor
	install -Dm755 reactor $(DESTDIR)/opt/llm-reactor/reactor

clean:
	rm -f reactor reactor_c_preview.png
