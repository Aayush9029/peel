package ui

import (
	"fmt"
	"os"
)

const (
	Reset   = "\x1b[0m"
	Bold    = "\x1b[1m"
	Dim     = "\x1b[2m"
	Red     = "\x1b[31m"
	Green   = "\x1b[32m"
	Yellow  = "\x1b[33m"
	Blue    = "\x1b[34m"
	Magenta = "\x1b[35m"
	Cyan    = "\x1b[36m"
)

func IsTTY() bool {
	fi, err := os.Stdout.Stat()
	if err != nil {
		return false
	}
	return (fi.Mode() & os.ModeCharDevice) != 0
}

func Header(msg string) {
	if IsTTY() {
		fmt.Printf("%s%s%s\n", Bold, msg, Reset)
	} else {
		fmt.Println(msg)
	}
}

func Success(msg string) {
	if IsTTY() {
		fmt.Printf("%s✓%s %s\n", Green, Reset, msg)
	} else {
		fmt.Println("ok:", msg)
	}
}

func Status(msg string) {
	if IsTTY() {
		fmt.Printf("%s→%s %s\n", Cyan, Reset, msg)
	} else {
		fmt.Println(msg)
	}
}

func Dimf(format string, args ...any) {
	if IsTTY() {
		fmt.Printf("%s"+format+"%s\n", append([]any{Dim}, append(args, Reset)...)...)
	} else {
		fmt.Printf(format+"\n", args...)
	}
}

func Errorf(format string, args ...any) {
	if IsTTY() {
		fmt.Fprintf(os.Stderr, "%s✗%s "+format+"\n", append([]any{Red, Reset}, args...)...)
	} else {
		fmt.Fprintf(os.Stderr, "error: "+format+"\n", args...)
	}
}

func Fatalf(format string, args ...any) {
	Errorf(format, args...)
	os.Exit(1)
}
