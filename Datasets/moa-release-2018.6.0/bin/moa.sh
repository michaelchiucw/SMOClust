#!/bin/bash

BASEDIR=`dirname $0`/..
BASEDIR=`(cd "$BASEDIR"; pwd)`
MEMORY=512m
java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2018.6.1-SNAPSHOT.jar:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.gui.GUI

