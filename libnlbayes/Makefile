INCDIR=include
SRCDIR=src
LIBDIR=lib

CFLAGS=-Wall -O3 -fopenmp -fPIC -shared -I$(INCDIR)
LIBS=-lgsl -lgslcblas -lm

INC=$(wildcard $(INCDIR)/*.h)
SRC=$(wildcard $(SRCDIR)/*.cpp)
LIB=$(LIBDIR)/libnlbayes.so

$(LIB):$(SRC) $(INC)
	mkdir -p $(LIBDIR)
	g++ $(SRC) -o $@ $(CFLAGS) $(LIBS)


ifeq ($(PREFIX),)
    PREFIX := /usr
endif

install:$(LIB)
	install -d $(DESTDIR)$(PREFIX)/lib/nlbayes
	install -m 644 $(LIB) $(DESTDIR)$(PREFIX)/lib/nlbayes
	install -d $(DESTDIR)$(PREFIX)/include/nlbayes
	install -m 644 $(INC) $(DESTDIR)$(PREFIX)/include/nlbayes

.PHONY: clean purge
clean :
	rm -rf $(LIBDIR)
purge :
	rm -rf $(LIBDIR)
