import sys, os, types

# Mock pygame ANTES de cualquier import
mp = types.ModuleType('pygame')
mp.__version__ = '0.0.0'
mp.init = lambda *a,**kw: (0,0)
mp.quit = lambda *a,**kw: None
mp.display = types.ModuleType('pygame.display')
mp.display.init = lambda *a,**kw: None
mp.display.set_mode = lambda *a,**kw: type('S',(),{'blit':lambda s,*a,**kw:None,'convert':lambda s:None})()
mp.display.set_caption = lambda *a,**kw: None
mp.display.quit = lambda *a,**kw: None
mp.event = types.ModuleType('pygame.event')
mp.event.pump = lambda *a,**kw: None
mp.image = types.ModuleType('pygame.image')
mp.image.load = lambda *a,**kw: type('S',(),{'blit':lambda s,*a,**kw:None,'convert':lambda s:None})()
mp.transform = types.ModuleType('pygame.transform')
mp.transform.scale = lambda *a,**kw: type('S',(),{'blit':lambda s,*a,**kw:None,'convert':lambda s:None})()
mp.surfarray = types.ModuleType('pygame.surfarray')
mp.surfarray.pixels3d = lambda *a,**kw: __import__('numpy').zeros((100,100,3),dtype='uint8')
mp.time = types.ModuleType('pygame.time')
mp.time.Clock = type('C',(),{'tick':lambda s,*a:None})
mp.font = types.ModuleType('pygame.font')
mp.font.init = lambda *a,**kw: None
mp.font.Font = type('F',(),{'render':lambda s,*a,**kw: type('S',(),{'blit':lambda s,*a,**kw:None,'convert':lambda s:None})()})
mp.Surface = type('S',(),{'blit':lambda s,*a,**kw:None,'convert':lambda s:None})
sys.modules['pygame'] = mp
for k in ['display','event','image','transform','surfarray','time','font']:
    sys.modules[f'pygame.{k}'] = getattr(mp, k)

from nbformat import read
from nbconvert.preprocessors import ExecutePreprocessor
import os

os.chdir(r'D:\ReforcingLearning\rl-basics')
src_path = r'D:\ReforcingLearning\rl-basics\notebooks\RL00_Problem_Formulation_FILLED.ipynb'
out_path = r'D:\ReforcingLearning\rl-basics\notebooks\RL00_Problem_Formulation_EXECUTED.ipynb'

with open(src_path, encoding='utf-8') as f:
    nb = read(f, as_version=4)

ep = ExecutePreprocessor(timeout=300, kernel_name='python3')
ep.preprocess(nb, {'metadata': {'path': os.path.dirname(src_path)}})

with open(out_path, 'w', encoding='utf-8') as f:
    from nbformat import write
    write(nb, f)
print(f'[OK] Notebook ejecutado guardado en: {out_path}')
