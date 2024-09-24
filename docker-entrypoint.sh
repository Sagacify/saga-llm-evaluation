#!/bin/sh

umask ${UMASK:-002}

exec "$@"