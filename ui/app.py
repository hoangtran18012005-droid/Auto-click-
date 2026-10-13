import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import cv2
import mss
import numpy as np
import yaml
from pynput import keyboard
from bot.runtime import Runtime
from bot.screen_capture import create_mss

class App:
    def __init__(self,root,config_path=None):
        self.root=Path(root)
        self.config_path=Path(config_path) if config_path else self.root/'config.yaml'
        self.cfg=yaml.safe_load(self.config_path.read_text(encoding='utf-8'))
        self.win=tk.Tk()
        self.win.title('Cuộc đua Kỳ Thú — Vision Auto Player')
        self.win.geometry('560x640')
        self.win.configure(bg='#eef4f5')
        self.bot=Runtime(self.cfg,self.root)
        self.pending=[]
        self.lock=threading.Lock()
        self.pressed=set()
        style=ttk.Style()
        style.theme_use('clam')
        style.configure('TButton',padding=8,font=('Segoe UI',10))
        panel=ttk.Frame(self.win,padding=20)
        panel.pack(fill='both',expand=True)
        ttk.Label(panel,text='VISION AUTO PLAYER',font=('Segoe UI',18,'bold')).pack(anchor='w')
        ttk.Label(panel,text='Ảnh game → dự đoán → tự nhảy').pack(anchor='w',pady=(3,14))
        row=ttk.Frame(panel);row.pack(fill='x')
        ttk.Button(row,text='Chọn vùng game',command=self.region).pack(side='left')
        ttk.Button(row,text='Lưu cấu hình',command=self.save).pack(side='left',padx=5)
        ttk.Button(row,text='Thông số',command=self.settings).pack(side='left')
        ttk.Label(panel,text='Mẫu ảnh: chọn trên ảnh game đã chuẩn hóa; Enter xác nhận.').pack(anchor='w',pady=(12,4))
        samples=ttk.Frame(panel);samples.pack(fill='x')
        for name,label in [('bird','Nhân vật'),('end','Kết thúc'),('ready','Sẵn sàng')]:
            ttk.Button(samples,text=label,command=lambda n=name:self.template(n)).pack(side='left',padx=2)
        controls=ttk.Frame(panel);controls.pack(fill='x',pady=15)
        ttk.Button(controls,text='F6 Bắt đầu / Tạm dừng',command=self.start).pack(side='left')
        ttk.Button(controls,text='F8 DỪNG',command=self.bot.stop).pack(side='left',padx=7)
        ttk.Button(panel,text='F7 Debug',command=self.debug).pack(anchor='w')
        ttk.Button(panel,text='Xuất 3 phiên log gần nhất',command=self.export_logs).pack(anchor='w',pady=5)
        self.text=tk.StringVar()
        ttk.Label(panel,textvariable=self.text,font=('Segoe UI',10),wraplength=470).pack(anchor='w',pady=15)
        ttk.Label(panel,text='Sau F6: chuyển về tab game trong 3 giây.\nỞ SẴN SÀNG, bot tự click VÀO ĐUA khi nhận đúng mẫu.\nF6 tạm dừng click; F8 dừng. Điểm “?” = chưa đọc được.\nOCR đọc điểm khi kết thúc; tự chơi lại là tùy chọn.',wraplength=500).pack(anchor='w')
        self.win.protocol('WM_DELETE_WINDOW',self.close)
        self.listener=keyboard.Listener(on_press=self.hotkey,on_release=self.key_release)
        self.listener.start()
        self.win.after(100,self.tick)

    def hotkey(self,key):
        if key not in (keyboard.Key.f6,keyboard.Key.f7,keyboard.Key.f8):return
        with self.lock:
            if key in self.pressed:return
            self.pressed.add(key)
        action={keyboard.Key.f6:self.start,keyboard.Key.f7:self.debug,keyboard.Key.f8:self.bot.stop}.get(key)
        if key==keyboard.Key.f8:
            with self.lock:self.pending.clear()
            self.bot.stop()
        elif action:
            with self.lock:
                self.pending.append((action,self.bot.generation))

    def key_release(self,key):
        with self.lock:self.pressed.discard(key)

    def save(self):
        self.config_path.write_text(yaml.safe_dump(self.cfg,sort_keys=False),encoding='utf-8')

    def settings(self):
        if self.bot.thread and self.bot.thread.is_alive():
            messagebox.showinfo('Thông số','Nhấn F8 và chờ bot dừng trước khi sửa thông số.');return
        dialog=tk.Toplevel(self.win);dialog.title('Thông số bot');dialog.transient(self.win)
        fields=[('fps','Giới hạn FPS',1),('gravity','Trọng lực (px/s²)',1),('jump','Vận tốc nhảy (px/s)',1),('pipe_speed','Tốc độ cột (px/s)',1),('latency','Trễ input (ms)',1000),('cooldown','Khoảng cách click (ms)',1000),('world_floor_px','Biên dưới ở ảnh rộng 600',1),('hitbox_half_height','Nửa cao hitbox; trống = bbox',1)]
        if self.cfg.get('planner')=='source_euler':
            fields.extend([('source_canvas_width','Canvas native rộng',1),('source_canvas_height','Canvas native cao',1),('source_ground_y','groundY từ JavaScript',1),('source_frame_dt','Bước vật lý (ms)',1000)])
        entries={}
        for row,(key,label,scale) in enumerate(fields):
            ttk.Label(dialog,text=label).grid(row=row,column=0,sticky='w',padx=10,pady=4)
            variable=tk.StringVar(value='' if self.cfg.get(key) is None else f'{self.cfg[key]*scale:.12g}')
            ttk.Entry(dialog,textvariable=variable,width=18,state='readonly' if key=='source_frame_dt' else 'normal').grid(row=row,column=1,padx=10);entries[key]=(variable,scale)
        planner=tk.StringVar(value=self.cfg.get('planner','maxmin'))
        ttk.Label(dialog,text='Planner; reachability là thử nghiệm').grid(row=len(fields),column=0,sticky='w',padx=10)
        ttk.Combobox(dialog,textvariable=planner,values=['maxmin','reachability','source_euler'],state='readonly',width=18).grid(row=len(fields),column=1,padx=10)
        restart=tk.BooleanVar(value=self.cfg.get('auto_restart',False))
        ttk.Checkbutton(dialog,text='Tự click CHƠI LẠI',variable=restart).grid(row=len(fields)+1,column=0,columnspan=2,sticky='w',padx=10,pady=8)
        def apply():
            try:
                updates={key:(None if not variable.get().strip() and key=='hitbox_half_height' else float(variable.get())/scale) for key,(variable,scale) in entries.items()}
                if not 10<=updates['fps']<=240 or updates['gravity']<=0 or updates['jump']>=0 or updates['pipe_speed']<=0 or not 0<=updates['latency']<=.5 or updates['cooldown']<.03 or updates['world_floor_px']<=0:raise ValueError('Thông số vật lý/FPS không hợp lệ.')
                if updates['hitbox_half_height'] is not None and updates['hitbox_half_height']<=0:raise ValueError('Nửa cao hitbox phải dương.')
                updates.update(planner=planner.get(),auto_restart=restart.get(),auto_start=True,input_mode='click',input_effect_latency=updates['latency'])
                candidate=dict(self.cfg,**updates)
                if candidate.get('planner')=='source_euler':
                    from bot.source_model import configure
                    configure(candidate)
                self.cfg.update(candidate);self.save();dialog.destroy()
            except ValueError as error:messagebox.showerror('Thông số',str(error),parent=dialog)
        ttk.Button(dialog,text='Lưu',command=apply).grid(row=len(fields)+2,column=0,columnspan=2,pady=10)

    def export_logs(self):
        if self.bot.thread and self.bot.thread.is_alive():
            messagebox.showinfo('Log','Nhấn F8 và chờ ghi log xong rồi xuất ZIP.');return
        if any(thread.is_alive() for thread in self.bot.score_threads):
            messagebox.showinfo('Log','OCR đang đọc điểm; chờ hoàn tất rồi xuất ZIP.');return
        from tools.export_logs import export_recent
        try:
            path=export_recent(self.root,3)
            messagebox.showinfo('Log',f'Đã tạo:\n{path}\nGửi ZIP cùng điểm thực tế và mô tả lỗi.')
        except Exception as error:messagebox.showerror('Log',str(error))

    def start(self):
        try:
            self.bot.start()
            if self.bot.thread and self.bot.thread.is_alive() and not self.bot.quit.is_set() and (not self.bot.pause.is_set() or self.bot.state.status.startswith('CHUYỂN VỀ TAB GAME')):
                self.win.iconify()
        except Exception as e:
            messagebox.showerror('Auto Player',str(e))

    def debug(self):
        self.bot.debug=not self.bot.debug
        if not self.bot.debug:
            cv2.destroyAllWindows()

    def region(self):
        self.bot.stop()
        self.win.withdraw()
        self.win.after(500,self._region)

    def _region(self):
        with create_mss() as sct:
            monitor=sct.monitors[0]
            frame=np.array(sct.grab(monitor))[:,:,:3]
        x,y,w,h=cv2.selectROI('Select complete canvas - Enter',frame,False,False)
        cv2.destroyAllWindows()
        if w>50 and h>50:
            self.cfg['roi']={'left':int(x)+monitor['left'],'top':int(y)+monitor['top'],'width':int(w),'height':int(h)}
            self.cfg['expected_window_rect']=None
            self.save()
        self.win.deiconify()

    def template(self,name):
        if not self.cfg.get('roi'):
            messagebox.showinfo('Mẫu','Chọn vùng game trước.');return
        self.bot.stop()
        self.win.withdraw()
        self.win.after(500,lambda:self._template(name))

    def _template(self,name):
        with create_mss() as sct:
            raw=np.array(sct.grab(self.cfg['roi']))[:,:,:3]
        frame=cv2.resize(raw,(self.cfg['width'],round(raw.shape[0]*self.cfg['width']/raw.shape[1])))
        x,y,w,h=cv2.selectROI(f'Sample {name} - Enter',frame,False,False)
        cv2.destroyAllWindows()
        if w and h:
            cv2.imwrite(str(self.root/'templates'/f'{name}.png'),frame[y:y+h,x:x+w])
            if name=='bird':
                self.cfg['bird_mode']='template'
            self.save()
        self.win.deiconify()

    def tick(self):
        with self.lock:
            actions=self.pending[:];self.pending.clear()
            for action,generation in actions:
                if generation==self.bot.generation:action()
        s=self.bot.state
        p=self.bot.model
        self.text.set(f'{s.status}\nFPS: {s.fps:.1f} | Thời gian lượt: {s.duration:.1f}s\nLượt kích hoạt: {s.runs} | Điểm ảnh: {s.score if s.score is not None else "?"}\nLượt trước (OCR): {s.last_result if s.last_result is not None else "?"} ({s.score_status})\nCao nhất đọc được: {s.best if s.best is not None else "?"}'+(f'\nv={p.v:.0f} | g={p.g:.0f} | tốc độ cột={p.speed:.0f}' if p else '')+f'\nLog: {Path(s.last_log).name if s.last_log else "chưa có"}')
        if self.bot.debug and self.bot.preview is not None:
            cv2.imshow('Debug (F7 hide)',self.bot.preview)
            cv2.waitKey(1)
        self.win.after(80,self.tick)

    def close(self):
        self.bot.stop()
        if self.bot.thread:
            self.bot.thread.join(timeout=2)
        self.save()
        self.listener.stop()
        cv2.destroyAllWindows()
        self.win.destroy()

    def run(self):
        self.win.mainloop()
