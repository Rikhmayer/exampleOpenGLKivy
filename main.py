
from kivy.app import App
from kivy.graphics.opengl import *
from kivy.graphics.opengl_utils import *

from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.uix.textinput import TextInput

from kivy.base import EventLoop
from kivy.clock import Clock

SceneWH = (300, 200) #тут не используется, но вдруг

import numpy as np
import time
from math import *
import pickle

from itertools import product

vertices  = np.zeros(20000*3, dtype=np.float32)
normals  = np.zeros(20000*3, dtype=np.float32)
texcoords = np.zeros(20000*2, dtype=np.float32)

ifVert = False
shaderProgram = glCreateProgram()
numVertsDyrty = 0

vertSH = '''
//vertex shaider

#ifdef GL_ES
    precision highp float;
#endif

attribute vec3 vPosition;
attribute vec3 vNormal;
attribute vec2 vTexcoords;

uniform mat4 PerspxViewMatr;
uniform mat4 NormRotMatr;

varying float LightI;
varying vec2 f2Texcoords;

void main()
{
    float ambient = 0.25;
    vec4 s = vec4(0.5, 0.5, 0.5, 0);

    gl_Position = PerspxViewMatr * vec4(vPosition[0], vPosition[1], vPosition[2], 1.0);

    vec4 norm = NormRotMatr * vec4(vNormal, 0);

    float dotnorm = dot(s, norm);
    LightI = ambient + max(dotnorm, -dotnorm)*0.65; //vec3(0.45, 0.45, 0.45);
    f2Texcoords = vTexcoords;
}
'''

fragSH = '''
//pixel shaider

#ifdef GL_ES
    precision highp float;
#endif

varying float LightI;
varying vec2 f2Texcoords;

uniform sampler2D textureObj;

void main()
{

    vec4 color = texture2D(textureObj, f2Texcoords);
    if(color[3] > 0.5)
        gl_FragColor = vec4( LightI*color.xyz, 1.0);
    else
        discard;

}
'''

def NormVec(v1):
    R = sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
    return [v1[0]/R, v1[1]/R, v1[2]/R]

def VecMult(v1, v2):
    return [ v1[1]*v2[2] - v1[2]*v2[1], v1[2]*v2[0] - v1[0]*v2[2], v1[0]*v2[1] - v1[1]*v2[0] ]

def MinusVecs(v1, v2):
    return [v1[0]-v2[0] , v1[1]-v2[1] , v1[2]-v2[2]]

def PlusVecs(v1, v2):
    return [v1[0]+v2[0] , v1[1]+v2[1] , v1[2]+v2[2]]

def CxV(C, v):
    return [C*v[0] , C*v[1] , C*v[2]]

class androidman():
    def __init__(self):
        self.V = []
        self.N = []
        self.T = []
        self.tfomx = 0.0
        self.tfomy = 0.0
        self.ttox = 1.0
        self.ttoy = 1.0

        self.coolbochka((0,0,0), (0,0,300), 200, 12)

        self.mpV = np.array(self.V, dtype = np.float32)
        self.mpN = np.array(self.V, dtype = np.float32)
        self.mpT = np.array(self.V, dtype = np.float32)
        self.VN = len(self.V)
        self.TN = len(self.T)
        self.shift = (0,0,0)

        #self.numVertsDyrty = int(len(self.V)/3)

    def OutVerts(self, V, N, T, Vrfom, Tfrom):
        '''перенесем свои значения векторов в глобальную систему векторов (не забудем повернуть, потом забудем!)'''

        for i in range(0, self.VN):
            V[Vrfom] = self.npV[i] + self.shift[i%3]
            N[Vrfom] = self.npN[i]
            Vrfom += 1
        for i in range(0, self.TN):
            T[Tfrom] = self.npT[i]
            Tfrom += 1

        self.ifChage = False
        return (Vrfom, Tfrom)

    def coolbochka(self, v1, v2, R, N, iye = False):
        ''' колбочка из точки в1 в точку в2 вот такой ширины, вот такой толщины, вот такой кривизны или с глазом '''
        alpha = 2*pi/N
        beta = alpha
        texstandart = [self.tfomx, self.tfomy, self.tfomx+self.ttox, self.tfomy, self.tfomx+self.ttox, self.tfomy+self.ttoy]

        #найдем вектор, перпендикулярный v1-v2
        dv = (v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]) #v1-v2
        NSphere = int(N/4)-1
        if NSphere < 2:
            NSphere = 2
        e = 1 #типа эксцентриситет
        normdno = CxV(-1, NormVec(dv))
        if abs(dv[2]) < 0.00001:
            radv = NormVec(VecMult(dv, (0,0,1)))
        else:
            radv = NormVec(VecMult(dv, (0,1,0)))
        for i in range(0, N-1):
            nextradv = NormVec(VecMult(dv, radv ))
            nextradv = ( radv[0]*cos(alpha) + nextradv[0]*sin(alpha), radv[1]*cos(alpha) + nextradv[1]*sin(alpha), radv[2]*cos(alpha) + nextradv[2]*sin(alpha) )

            self.V += ( PlusVecs(CxV(R, radv), v1 ), PlusVecs(CxV(R, nextradv), v1 ), PlusVecs(CxV(R, nextradv), v2 ) )
            self.V += ( PlusVecs(CxV(R, nextradv), v2 ), PlusVecs(CxV(R, radv), v2 ), PlusVecs(CxV(R, radv), v1 ) )
            self.N += radv + nextradv + nextradv + nextradv + radv + radv
            self.T += texstandart*2
            #и донышко закроем
            self.V += ( PlusVecs(CxV(R, radv), v1 ), PlusVecs(CxV(R, nextradv), v1 ), v1 )
            self.N += normdno*3
            self.T += texstandart
            #и круглую часть
            D0, D1 = radv, nextradv
            N0, N1 = radv, nextradv
            for j in range(0, NSphere):
                D2 = PlusVecs( CxV(cos((j+1)*beta), PlusVecs(CxV(R, nextradv), v2 )),  CxV(-e*R*sin((j+1)*beta),normdno))
                D3 = PlusVecs( CxV(cos((j+1)*beta), PlusVecs(CxV(R, radv), v2 )),  CxV(-e*R*sin((j+1)*beta),normdno))
                N2 = NormVec(PlusVecs( CxV(cos((j+1)*beta), nextradv),  CxV(-e*sin((j+1)*beta),normdno)))
                N3 = NormVec(PlusVecs( CxV(cos((j+1)*beta), radv),  CxV(-e*sin((j+1)*beta),normdno)))
                self.V += D0+D1+D2  +  D2+D3+D0
                self.N += N0+N1+N2  +  N2+N3+N0
                self.T += texstandart*2
                D0, D1 = D2, D3
            #и пимпочка сверху
            self.V += D0+D1+ PlusVecs(CxV(-e*R,normdno), v2 ))
            self.N += N0+N1+CxV(-1, normdno)
            self.T += texstandart
            #переходим к следующей дольке
            radv = nextradv

def MatrixViev():
    #делаем матрицу вида как показано в мануале http://www.songho.ca/opengl/gl_projectionmatrix.html
    f = 5500#3000#типа дальняя граница

    r = SceneWH[0]
    t = SceneWH[1]

    n0, r0, t0 = 1500, 750, 250 #это базовые настройки разрешения, на которые я ориентируюсь
    n = 1500

    if ifVert:
        r = 300
        t = r*SceneWH[1]/SceneWH[0]
    else:
        t = 300
        r = t*SceneWH[0]/SceneWH[1]

    alpha = math.pi/2 - math.asin(1/math.sqrt(2))#math.pi/6
    F = 3000#2000 #расстояние от фокуса до точвки 0 0 0

    perspective = [[n/r , 0, 0, 0],
                   [ 0, n/t, 0, 0],
                   [ 0, 0, -(f + n)/(f - n), -2*f*n/(f - n)],
                   #[ 0, 0, -2/(f - n), -(f + n)/(f - n)], #ортогональная проекция
                   #[ 0, 0, 0, 1], #ортогональная проекция
                   [ 0, 0, -1, 0]]
    if ifVert:
        rotaxis     = [[ 0, math.cos(alpha), math.sin(alpha), 0],
                      [ -1, 0, 0, 0], #[ 1, 0, 0, 0],
                   [ 0, -math.sin(alpha), math.cos(alpha), 0],
                   [ 0, 0, 0, 1]]
    else:
        rotaxis     = [[ 1, 0, 0, 0],
                   [ 0, math.cos(alpha), math.sin(alpha), 0],
                   [ 0, -math.sin(alpha), math.cos(alpha), 0],
                   [ 0, 0, 0, 1]]
    #print(np.dot(rotaxis, (0,1,0,1)))
    shiftaxis   = [[ 1, 0, 0, 0],
                   [ 0, 1, 0, 0],
                   [ 0, 0, 1, -F],
                   [ 0, 0, 0, 1]]
    return np.dot( np.array(perspective, dtype=np.float32), np.dot( np.array(shiftaxis, dtype=np.float32), np.array(rotaxis, dtype=np.float32)))

def MatrixRot():
    alpha = math.pi/2 - math.asin(1/math.sqrt(2))
    if ifVert:
        rotaxis     = [[ 0, math.cos(alpha), math.sin(alpha), 0],
                      [ -1, 0, 0, 0],
                   [ 0, -math.sin(alpha), math.cos(alpha), 0],
                   [ 0, 0, 0, 1]]
    else:
        rotaxis     = [[ 1, 0, 0, 0],
                   [ 0, math.cos(alpha), math.sin(alpha), 0],
                   [ 0, -math.sin(alpha), math.cos(alpha), 0],
                   [ 0, 0, 0, 1]]
    return np.array(rotaxis, dtype=np.float32)

def GenTexture():
    txtr = np.zeros((100*100*3), dtype = np.int8)
    for i in range(0,100):
        for j in range(0,100):
            txtr[(i*100+j)*3] = 30
            txtr[(i*100+j)*3+1] = 220
            txtr[(i*100+j)*3+3] = 50
    return txtr.tobytes()

def init(): #Window):
        #global SceneWH
        #SceneWH = (int(Window.width/2), int(Window.height/2))

        texID2 = glGenTextures(1)
        texID = texID2[0]

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texID)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 3)
        glPixelStorei(GL_PACK_ALIGNMENT, 3)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        #print(len(textr[2]), textr[0]*textr[1]*3, textr[0], textr[1] )
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 100, 100, 0, GL_RGB, GL_UNSIGNED_BYTE, GenTexture())

        VshaderID = glCreateShader( GL_VERTEX_SHADER )#shader)
        glShaderSource(VshaderID, vertSH.encode('utf-8'))#getFileContent("helloTriangle.vert"))

        FshaderID = glCreateShader( GL_FRAGMENT_SHADER )#shader)
        glShaderSource(FshaderID, fragSH.encode('utf-8'))#getFileContent("helloTriangle.frag"))

        glCompileShader(VshaderID)
        glCompileShader(FshaderID)
        glAttachShader(shaderProgram, VshaderID )
        glAttachShader(shaderProgram, FshaderID )
        glLinkProgram(shaderProgram)

        glUseProgram(shaderProgram)
        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glEnableVertexAttribArray(2)

        glUseProgram(shaderProgram)
        ViewMatrLocation = glGetUniformLocation(shaderProgram, b'PerspxViewMatr')

        glUniformMatrix4fv(ViewMatrLocation, 1 , GL_FALSE , bytes ( MatrixViev().transpose() ) )

        RotMatrLocation = glGetUniformLocation(shaderProgram, b'NormRotMatr')

        glUseProgram(shaderProgram)
        glUniformMatrix4fv(  RotMatrLocation, 1 , GL_FALSE , bytes ( MatrixRot().transpose() ) )

        glEnable(GL_DEPTH_TEST)

init()

AM = androidman()

class CustomWidget(Widget):

    def __init__(self, **kwargs):
        super(CustomWidget, self).__init__(**kwargs)

        self.flagTochingKofsh = False
        self.ReloadVertewxes = True

    def on_touch_move(self, touch):#on_touch_move(self, touch):
        #print("numVertsDyrty = ", numVertsDyrty, touch)
        pos = ( touch.spos[0]*2-1 , touch.spos[1]*2-1  )
        self.pos = pos
        self.ReloadVertewxes = True

    def on_touch_down(self, touch):
        #print(touch.spos[0], touch.spos[1]) # - координаты от 0 до 1, где 0, 0 - левый нижний угол
        pos = ( touch.spos[0]*2-1 , touch.spos[1]*2-1  )
        self.pos = pos

    def update_glsl(self, nap):
        global numVertsDyrty
        global vertices3
        global normals3
        global texcoords3

        #numVertsDyrty =  int(lv/3)

        if self.ReloadVertewxes:
            vertices3 = vertices.tobytes('C')
            normals3 = normals.tobytes('C')
            texcoords3 = texcoords.tobytes('C') #а вот они, кстати, не меняются при некоторой аккуратности
            glUseProgram(shaderProgram)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0,  vertices3 ) #а здесь можно попробовать на с 0 менять куски векторов
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0,  normals3 ) #.tobytes('C') )
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0,  texcoords3 )
            self.ReloadVertewxes = False

        glClearColor(0.25, 0.25, 0.25, 1.0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        glDrawArrays(GL_TRIANGLES, 0, numVertsDyrty)

        Window.flip()

    def on_touch_up(self, touch):
        return

def passFunc(W):
    pass

class MainApp(App):
    def build(self):
        root = CustomWidget()
        EventLoop.ensure_window() #?
        Window.on_flip = lambda W = Window: passFunc(W)

        return root

    def on_start(self):

        global numVertsDyrty
        (lv, lt) = AM.OutVerts(vertices, normals, texcoords, 0, 0)
        numVertsDyrty =  int(lv/3)
        global vertices3
        global normals3
        global texcoords3

        vertices3 = vertices.tobytes('C')
        normals3 = normals.tobytes('C')
        texcoords3 = texcoords.tobytes('C')
        Clock.schedule_interval(self.root.update_glsl, 20 ** -1)

if __name__ == '__main__':
    MainApp().run()
