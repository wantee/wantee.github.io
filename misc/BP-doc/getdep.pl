#/usr/bin/env perl

while(<>) {
    if (m/\\input\{(.*)\}/) {
        print "$1.tex\n";
    }
}
