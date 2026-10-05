"""One-window converter GUI (tkinter, ships with Python)."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import traceback

from . import __version__


def main(vrm_path: str = "", game_dir: str = "") -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError:
        print("tkinter is not available; use the command line instead: python -m vroid2diva --help")
        return 1

    from .convert.builder import BuildOptions
    from .convert.game import find_game_dir, is_game_dir
    from .convert.pipeline import ConvertRequest, convert, list_modules

    root = tk.Tk()
    root.title("VRoid → Project DIVA Mega Mix+  (v%s)" % __version__)
    root.minsize(640, 560)
    pad = {"padx": 8, "pady": 4}

    vrm_var = tk.StringVar(value=vrm_path)
    game_var = tk.StringVar(value=game_dir or find_game_dir() or "")
    name_var = tk.StringVar(value=os.path.splitext(os.path.basename(vrm_path))[0] if vrm_path else "")
    module_var = tk.StringVar()
    filter_var = tk.StringVar()
    physics_var = tk.BooleanVar(value=True)
    face_var = tk.BooleanVar(value=True)
    toon_var = tk.BooleanVar(value=True)
    size_var = tk.StringVar(value="2048")
    status_var = tk.StringVar(value="")
    modules = []
    log_q: "queue.Queue" = queue.Queue()

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="both", expand=True)
    frm.columnconfigure(1, weight=1)

    def browse_vrm():
        p = filedialog.askopenfilename(title="Choose your VRoid model",
                                       filetypes=[("VRM model", "*.vrm"), ("All files", "*.*")])
        if p:
            vrm_var.set(p)
            if not name_var.get():
                name_var.set(os.path.splitext(os.path.basename(p))[0])

    def browse_game():
        p = filedialog.askdirectory(title="Choose the folder that contains DivaMegaMix.exe")
        if p:
            game_var.set(p)
            load_modules()

    def load_modules():
        modules.clear()
        combo["values"] = []
        g = game_var.get()
        if not is_game_dir(g):
            status_var.set("Select the Mega Mix+ folder (the one containing DivaMegaMix.exe).")
            return
        status_var.set("Reading module list ...")
        root.update_idletasks()
        try:
            modules.extend(list_modules(g, log=lambda s: None))
        except Exception as e:
            status_var.set("Could not read the game's modules: %s" % e)
            return
        apply_filter()
        status_var.set("%d modules found. Pick one to replace with your VRoid model." % len(modules))

    def apply_filter(*_):
        f = filter_var.get().lower()
        combo["values"] = [m.label() for m in modules if f in m.label().lower()]

    ttk.Label(frm, text="1. VRoid model (.vrm)").grid(row=0, column=0, sticky="w", **pad)
    ttk.Entry(frm, textvariable=vrm_var).grid(row=0, column=1, sticky="ew", **pad)
    ttk.Button(frm, text="Browse…", command=browse_vrm).grid(row=0, column=2, **pad)

    ttk.Label(frm, text="2. Game folder").grid(row=1, column=0, sticky="w", **pad)
    ttk.Entry(frm, textvariable=game_var).grid(row=1, column=1, sticky="ew", **pad)
    ttk.Button(frm, text="Browse…", command=browse_game).grid(row=1, column=2, **pad)

    ttk.Label(frm, text="3. Module to replace").grid(row=2, column=0, sticky="w", **pad)
    combo = ttk.Combobox(frm, textvariable=module_var, state="readonly")
    combo.grid(row=2, column=1, sticky="ew", **pad)
    search = ttk.Entry(frm, textvariable=filter_var, width=14)
    search.grid(row=2, column=2, **pad)
    filter_var.trace_add("write", apply_filter)
    ttk.Label(frm, text="(search)", foreground="#777").grid(row=3, column=2, sticky="n")

    ttk.Label(frm, text="4. Name shown in game").grid(row=3, column=0, sticky="w", **pad)
    ttk.Entry(frm, textvariable=name_var).grid(row=3, column=1, sticky="ew", **pad)

    opt = ttk.LabelFrame(frm, text="Options", padding=6)
    opt.grid(row=4, column=0, columnspan=3, sticky="ew", **pad)
    ttk.Checkbutton(opt, text="Hair / skirt physics (spring bones)", variable=physics_var).grid(row=0, column=0, sticky="w")
    ttk.Checkbutton(opt, text="Facial expressions & lip sync (blend shapes)", variable=face_var).grid(row=1, column=0, sticky="w")
    ttk.Checkbutton(opt, text="Toon shading (DIVA character shaders)", variable=toon_var).grid(row=2, column=0, sticky="w")
    ttk.Label(opt, text="Max texture size").grid(row=0, column=1, sticky="e", padx=(30, 4))
    ttk.Combobox(opt, textvariable=size_var, values=["1024", "2048", "4096"], width=6,
                 state="readonly").grid(row=0, column=2, sticky="w")

    convert_btn = ttk.Button(frm, text="Convert and install mod")
    convert_btn.grid(row=5, column=0, columnspan=3, sticky="ew", **pad)
    bar = ttk.Progressbar(frm, maximum=1.0)
    bar.grid(row=6, column=0, columnspan=3, sticky="ew", **pad)
    ttk.Label(frm, textvariable=status_var, foreground="#225").grid(row=7, column=0, columnspan=3, sticky="w", **pad)
    log = tk.Text(frm, height=12, wrap="word", state="disabled")
    log.grid(row=8, column=0, columnspan=3, sticky="nsew", **pad)
    frm.rowconfigure(8, weight=1)

    def write_log(text):
        log.configure(state="normal")
        log.insert("end", text + "\n")
        log.see("end")
        log.configure(state="disabled")

    def poll():
        try:
            while True:
                kind, value = log_q.get_nowait()
                if kind == "log":
                    write_log(value)
                elif kind == "progress":
                    bar["value"] = value
                elif kind == "done":
                    convert_btn.state(["!disabled"])
                    status_var.set("Finished! Start the game and choose the module you replaced.")
                    if messagebox.askyesno("Done", "Your mod was created in:\n\n%s\n\nOpen the folder?" % value):
                        open_folder(value)
                elif kind == "error":
                    convert_btn.state(["!disabled"])
                    status_var.set("Conversion failed - see the log.")
                    messagebox.showerror("Conversion failed", value)
        except queue.Empty:
            pass
        root.after(100, poll)

    def start():
        if not os.path.isfile(vrm_var.get()):
            messagebox.showwarning("Missing model", "Please choose a .vrm file.")
            return
        if not is_game_dir(game_var.get()):
            messagebox.showwarning("Missing game", "Please choose the Mega Mix+ game folder.")
            return
        label = module_var.get()
        mod = next((m for m in modules if m.label() == label), None)
        if mod is None:
            messagebox.showwarning("Missing module", "Please choose which module to replace.")
            return
        opts = BuildOptions(style="toon" if toon_var.get() else "simple", physics=physics_var.get(),
                            face=face_var.get(), max_texture_size=int(size_var.get()))
        req = ConvertRequest(vrm_var.get(), game_var.get(), mod.id, name_var.get().strip(), options=opts)
        convert_btn.state(["disabled"])
        bar["value"] = 0
        log.configure(state="normal")
        log.delete("1.0", "end")
        log.configure(state="disabled")

        def work():
            try:
                out = convert(req, log=lambda s: log_q.put(("log", s)), progress=lambda f: log_q.put(("progress", f)))
                log_q.put(("done", out))
            except Exception as e:
                log_q.put(("log", traceback.format_exc()))
                log_q.put(("error", str(e)))

        threading.Thread(target=work, daemon=True).start()

    convert_btn.configure(command=start)
    if game_var.get():
        root.after(200, load_modules)
    else:
        status_var.set("Select your VRoid model and the Mega Mix+ game folder to begin.")
    root.after(100, poll)
    root.mainloop()
    return 0


def open_folder(path: str) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        pass
