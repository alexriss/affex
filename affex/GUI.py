# GUI for AffexApp
# so far it is rather experimental


import os
os.environ["KIVY_NO_ARGS"] = "1"
dir_path = os.path.dirname(os.path.realpath(__file__))

from kivy.config import Config
Config.set('kivy', 'log_level', 'warning')
Config.set('kivy', 'window_icon', os.path.join(dir_path,'icon.png'))
Config.set('graphics', 'width', '800')
Config.set('graphics', 'height', '700')
Config.set('kivy', 'exit_on_escape', '0')

from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import NumericProperty, ListProperty, ObjectProperty
from kivy.core.window import Window
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Line

import cv2
import threading
import affex

#os.environ['KIVY_VIDEO'] = 'ffpyplayer' #ffpyplayer'  # 'gstplayer'  # gstreamer should be installed




def frame_to_image(frame):
    buf = cv2.flip(frame, 0).tostring()
    image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
    image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
    return image_texture

ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n/10%10!=1)*(n%10<4)*n%10::4])

class HelperManager:
    def get_ordinal(self, num):
        num = int(num)
        if num==1:
            return ''
        else:
            return ordinal(int(num))

class StackMovieScreen(Screen):
    helper = HelperManager()
    app = None
    capture = None
    slider_markers = [None, None]
    popup_error = None
    load_path = None
    
    def __init__(self, *args, **kwargs):
        super(StackMovieScreen, self).__init__(*args, **kwargs)
        Window.bind(on_dropfile=self._on_file_drop)
        self.app = App.get_running_app()
        self.popup_error = PopupError()
        
        self.ids.first.bind(value=self._mark_first)
        self.ids.last.bind(value=self._mark_last)
        self.ids.video_slider.bind(size=self._mark_firstlast)
        
    def _mark_firstlast(self, first=True, last=True):
        if (not first) and (not last): return
        if first:
            marker = 0
            val = self.ids.first.value/100
        elif last:
            marker = 1
            val = self.ids.last.value/100
        s = self.ids.video_slider
        p = (s.x + s.padding + val * (s.width - 2 * s.padding), s.y)
        points = [p[0], p[1]+s.height/3, p[0], p[1]+s.height/3*2]
        if not self.slider_markers[marker]:
            with self.ids.video_slider.canvas:
                self.slider_markers[marker] = Line(points=points, width=2)
        else:
            self.slider_markers[marker].points = points
        if first and last:
            self._mark_firstlast(first=False, last=True)
        
    def _mark_first(self, *args):
        self._mark_firstlast(1)

    def _mark_last(self, *args):
        self._mark_firstlast(0)

    def _on_file_drop(self, window, file_path):
        self.load_file(file_path.decode("utf-8"))
        
    def dismiss_popup_load(self):
        self.popup_load.dismiss()
        
    def open_file_dialog(self):
        self._video_player_pause()
        self.app.root.transition.direction = 'right'
        self.app.root.current = 'stack_movie_screen'
        content = LoadDialog(load=self.load_file_init, cancel=self.dismiss_popup_load)
        if self.load_path:
            dir = self.load_path
        else:
            dir = os.getcwd()
        content.ids.filechooser.path = dir
        self.popup_load = Popup(title="Load video file...", content=content, size_hint=(0.9, 0.9))
        self.popup_load.open()
        
    def load_file_init(self, path, filename):
        self.load_file(os.path.join(path, filename))
        self.load_path = path
        self.dismiss_popup_load()
        
    def load_file(self, filename):
        print('Loading: %s' % filename)
        self.capture = cv2.VideoCapture(filename)
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        totalFrames = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps>0 and totalFrames>1:
            self.ids.button_process.disabled = False
            self.ids.image_video_player.init_update(capture=self.capture)
            self.capture_filename = filename
        else:
            self.unload_file()
            if not fps>0:
                txt_error = 'Cannot open file %s.\nUnsupported file format.' % os.path.basename(filename)
            else:
                txt_error = 'Frames in file %s: %d.\nWe need more to stack and align.' % (filename, totalFrames)
            self.popup_error.ids.label_error.text = txt_error
            self.popup_error.open()
            return False
    
    def unload_file(self):
        self.capture = None
        self.capture_filename = None
        self.ids.image_video_player.capture = None
        self.ids.image_video_player.texture = None
        self.ids.button_process.disabled = True
        self._video_player_pause()
        
    def _video_player_play(self):
        self.ids.image_video_player.start_updating()
        
    def _video_player_pause(self):
        self.ids.image_video_player.stop_updating()
        
    def on_stop(self):
        self.capture.release()     
    
    def stack_movie(self):
        if self.capture:
            self.app.root.transition.direction = 'left'
            self.app.root.transition.duration = 0.5
            self.app.root.current = 'result_screen'
            
            self.app.result_screen._init_stack_movie_calculate(self.capture)
        

class KivyCV(Image):
    def __init__(self, **kwargs):
        super(KivyCV, self).__init__(**kwargs)
        self.total_frames = 0
        self.current_frame = 0
        self.app = None
        self.capture = None
        self.player_clock = None
        self.fps = 1
        #Image.__init__(self, **kwargs)
        
    def _get_app(self):
        if not self.app:
            self.app = App.get_running_app()
        return self.app
            
    def _on_slider_value_change(self,instance,value):
        if self.capture:
            value = round(value)
            if value <= self.total_frames:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES,value)
                self.current_frame = value
                self.update(playing=False)

    def _slider_change(self, dx):
        app = self._get_app()
        val = app.stack_movie_screen.ids.video_slider.value
        val += dx
        val = max(val,app.stack_movie_screen.ids.video_slider.min)
        val = min(val,app.stack_movie_screen.ids.video_slider.max)
        app.stack_movie_screen.ids.video_slider.value = val

    def _on_touch_move(self, widget, touch):
        if 'alt' in Window.modifiers:
            if self.collide_point(touch.x, touch.y):
                self._slider_change(self.total_frames*touch.dsx)
        
    def _on_touch_down(self, widget, touch):
        if 'alt' in Window.modifiers:
            if self.collide_point(touch.x, touch.y):
                if touch.is_mouse_scrolling:
                    if touch.button == 'scrolldown':
                        self._slider_change(-4*touch.psy)
                    elif touch.button == 'scrollup':
                        self._slider_change(4*touch.psy)
        
    def init_update(self, capture=None, **kwargs):
        self.capture = capture
        if self.capture:
            try:
                self.fps = self.capture.get(cv2.CAP_PROP_FPS)
                self.total_frames = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
                self.current_frame = 0
                app = self._get_app()
                self.video_slider = app.stack_movie_screen.ids.video_slider
                self.video_slider.bind(value=self._on_slider_value_change)
                self.video_slider.max = self.total_frames
                self.video_slider.value = self.current_frame
                self.capture.set(cv2.CAP_PROP_POS_FRAMES,self.current_frame)
                self.start_updating()
            except:
                self._error_reading_file()
                
    def _error_reading_file(self):
        self.app.stack_movie_screen.unload_file()
        txt_error = 'Error reading file.'
        self.app.stack_movie_screen.popup_error.ids.label_error.text = txt_error
        self.app.stack_movie_screen.popup_error.open()           
        return False
            
    def start_updating(self):
        if self.capture:
            self.player_clock = Clock.schedule_interval(self.update, 1.0/self.fps)
    
    def stop_updating(self):
        if self.player_clock:
            self.player_clock.cancel()
            self.player_clock = None
    
    def update(self, dt=0, playing=True):
        try:
            ret, frame = self.capture.read()
            if ret:
                self.texture = frame_to_image(frame)
                self.video_slider.value = self.current_frame
                if playing and self.current_frame+1 < self.total_frames:
                    self.current_frame += 1
            else:
                self.stop_updating()
        except:
            self._error_reading_file()


            
class ResultScreen(Screen):
    def __init__(self, *args, **kwargs):
        super(ResultScreen, self).__init__(*args, **kwargs)
        self.app = App.get_running_app()
        self.process_info = {'calculating': False, 'out_progress': 0, 'in_stop': False, 'out_ended': False, 'in_use_temp_images': True, 'out_temp_images': []}
        self.capture = None
        self.images = [None] * 3
        self.waiter = None
        
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        
    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'escape':
           self._stop_stack_movie_calculate()
        if keycode[1] == "o" and 'ctrl' in modifiers:
            self.app.stack_movie_screen.open_file_dialog()
        return True
        
    def _init_stack_movie_calculate(self, capture):
        # need to stop running thread first
        if self.waiter: self.waiter.cancel()
        if self.process_info['out_ended']:
            self.process_info['calculating'] = False
        if self.process_info['calculating']:
            self.ids.label_status.text = 'stopping previous...'
            self.process_info['in_stop'] = True
            Clock.schedule_once(lambda dt: self._init_stack_movie_calculate(capture),0.2)
            return
        self.process_info['in_stop'] = False
        self.process_info['out_ended'] = False
        self.process_info['out_temp_images'] = []
        self.process_info['calculating'] = True
        self.app.result_screen.ids.result_save.disabled = True
        self.ids.image_result.opacity = 0
        self.capture = capture
        self.ids.progressbar.height = '50sp'
        self.ids.progressbar.opacity = 1
        self.ids.label_spacer.height = '250sp'
        self.ids.label_status.text = 'calculating...'
        
        self.thread_stack_movie_calculate = threading.Thread(target=self._stack_movie_calculate) # ,args=(q,)
        self.app.stack_movie_screen.ids.image_video_player.stop_updating()
        self.thread_stack_movie_calculate.start()
        # we need to do this, to get out of the thread
        self.waiter = Clock.schedule_interval(self._wait_to_finish, 0.2)
        
    def _wait_to_finish(self, *args):
        if not self.process_info['calculating']:
            self.waiter.cancel()
            self.show_image()
        else:
            if len(self.process_info['out_temp_images'])>0:
                self.images = self.process_info['out_temp_images']
                self.show_image(temporary=True)
            
        self.ids.progressbar.value = self.process_info['out_progress']
    
    def _stop_stack_movie_calculate(self):
        self.ids.label_status.text = 'stopping...'
        self.process_info['in_stop'] = True
    
    def _stack_movie_calculate(self):
        first = self.app.stack_movie_screen.ids.first.value/100
        last = self.app.stack_movie_screen.ids.last.value/100
        frame_step = self.app.stack_movie_screen.ids.frame_step.value
        align = self.app.stack_movie_screen.ids.align.active
        align_method = self.app.stack_movie_screen.ids.dropbut_align_method.text
        align_warp_mode = self.app.stack_movie_screen.ids.dropbut_align_warp_mode.text.lower()
        # get stacked image, first image, last image
        self.images = affex.stackMovie(self.capture, first=first, last=last, frame_step=frame_step, align=align, align_method=align_method, align_warp_mode=align_warp_mode, process_info=self.process_info)
        self.process_info['calculating'] = False

    def show_image(self, temporary=False):
        if not temporary:
            self.ids.progressbar.height = '0sp'
            self.ids.progressbar.opacity = 0
            if (self.process_info['in_stop'] == True):
                self.ids.label_status.text = 'Finished (prematurely).'
            else:
                self.ids.label_status.text = 'Finished.'
        self.ids.label_spacer.height = '0sp'
        self.ids.image_result.texture = frame_to_image(self.images[0])
        self.ids.image_result.opacity = 1
        self.ids.result_save.disabled = False
        self.app.stack_movie_screen.ids.button_previous_result.disabled = False
        

    def dismiss_popup_save(self):
        self.popup_save.dismiss()
        
    def save_image_overwrite(self,overwrite=False):
        self.popup_save_exists.dismiss()
        self.save_overwrite = overwrite
        if overwrite:
            self.save_image_write()
        
    def save_image_write(self):
        extension = os.path.splitext(self.save_filename)[1][1:]
        if not extension in ['png', 'jpg']:
            self.popup_save_error = PopupError()
            self.popup_save_error.ids.label_error.text = 'Unsupported extension: %s.' % extension
            self.popup_save_error.open()
        elif os.path.isfile(self.save_filename) and not self.save_overwrite:
            self.popup_save_exists = PopupFileExists()
            self.popup_save_exists.ids.label_text.text = 'File %s exists.\n\nOverwrite?' % self.save_filename
            self.popup_save_exists.open()
        else:
            print('Saving: %s' % self.save_filename)
            cv2.imwrite(self.save_filename, self.images[0], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if self.process_info['calculating']:
                self.ids.label_status.text = 'calculating... (saved temporary result).'
            else:
                self.ids.label_status.text = 'Saved.'
            self.dismiss_popup_save()
        
            
    def save_image_init(self, path, filename):
        self.save_filename  = os.path.join(path, filename)
        self.save_overwrite = False
        self.save_image_write()
   
    def save_image(self):
        if self.ids.result_save.disabled:
            return False
        dir = os.path.dirname(os.path.realpath(self.app.stack_movie_screen.capture_filename))
        content = SaveDialog(save=self.save_image_init, cancel=self.dismiss_popup_save)
        content.ids.filechooser.path = dir
        content.ids.text_input.text = os.path.splitext(self.app.stack_movie_screen.capture_filename)[0]+'.jpg'
        self.popup_save = Popup(title="Save image...", content=content, size_hint=(0.9, 0.9))
        self.popup_save.open()
        
    
    

class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    cancel = ObjectProperty(None)

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    
    
class PopupFileExists(Popup):
    overwrite=False
    
class PopupError(Popup):
    pass
    
    
class DropBut(Button):
    types = []
    selected = 0
    def __init__(self, **kwargs):
        super(DropBut, self).__init__(**kwargs)
        self.drop_list = DropDown()
        for i in self.types:
            btn = Button(text=i, size_hint_y=None, height=45)
            btn.bind(on_release=lambda btn: self.drop_list.select(btn.text))
            self.drop_list.add_widget(btn)

        self.bind(on_release=self.drop_list.open)
        self.drop_list.bind(on_select=lambda instance, x: setattr(self, 'text', x))
        if self.selected > -1 and self.selected < len(self.types):
            self.text = self.types[self.selected]
    
    
class DropButAlignMethod(DropBut):
    types = ['ORB', 'ECC']
    selected = 1 
    def __init__(self, **kwargs):
        super(DropButAlignMethod, self).__init__(**kwargs)
        
    
class DropButAlignWarpMode(DropBut):
    types = ['Translation', 'Euclidean', 'Affine', 'Homography']
    selected = 2
    def __init__(self, **kwargs):
        super(DropButAlignWarpMode, self).__init__(**kwargs)
    
    


class AffexApp(App):
    title = "Affex"
    filename_startup = None
    
    def __init__(self, parser_args={}, *args, **kwargs):
        App.__init__(self)
        if len(parser_args.filename)>0:
            self.filename_startup = parser_args.filename

    
    def build(self):
        self.stack_movie_screen = StackMovieScreen(name='stack_movie_screen')
        self.result_screen = ResultScreen(name='result_screen')
        self.screen_manager = ScreenManager()
        self.screen_manager.add_widget(self.stack_movie_screen)
        self.screen_manager.add_widget(self.result_screen)
        
        # testing code
        if False:
            self.screen_manager.current = 'result_screen'
            self.result_screen.ids.progressbar.value = 50
        if False:
            self.screen_manager.current = 'result_screen'
            self.result_screen.ids.progressbar.height = '0dp'
            self.result_screen.ids.canvas_spacer.height = '0dp'
            self.result_screen.ids.progressbar.opacity = 0

            capture = cv2.VideoCapture('2018-07-29 19.30.40.mp4')
            ret, frame = capture.read()
            #self.images = affex.stackMovie(capture, first=0.5, last=0.6, frame_step=5, align=True, align_method='ECC', align_warp_mode='euclidean')
            #self.result_screen.ids.image_result.texture = frame_to_image(self.images[0])
            self.result_screen.ids.image_result.texture = frame_to_image(frame)
    
        return self.screen_manager
        
    def on_start(self):
        if self.filename_startup:
            self.stack_movie_screen.load_file(self.filename_startup)
        
    def on_stop(self):
        self.result_screen.process_info['in_stop'] = True
        return True        