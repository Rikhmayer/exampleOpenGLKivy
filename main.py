
from kivy.app import App
from kivy.graphics.opengl import *

from kivy.uix.widget import Widget
from kivy.core.window import Window

from kivy.base import EventLoop
from kivy.clock import Clock

import numpy as np
from math import *

maxnumVertsDyrty = 10000
vertices  = np.zeros((maxnumVertsDyrty,3), dtype=np.float32)
normals  = np.zeros((maxnumVertsDyrty,3), dtype=np.float32)
texcoords = np.zeros(maxnumVertsDyrty*2, dtype=np.float32)

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

    vec4 norm = NormRotMatr * vec4(vNormal, 0.0);

    float dotnorm = dot(s, norm);
    LightI = ambient + max(dotnorm, 0.0)*0.65; //vec3(0.45, 0.45, 0.45);
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
    gl_FragColor = vec4( LightI*color.xyz, 1.0);

}
'''

class vector():
    def __init__(self, x, y, z):
        self.v = [x, y, z]
    def __add__(self, other):
        return vector(self.v[0] + other.v[0], self.v[1] + other.v[1], self.v[2] + other.v[2])
    def __sub__(self, other):
        return vector(self.v[0] - other.v[0], self.v[1] - other.v[1], self.v[2] - other.v[2])
    def __mul__(self, other): #векторное умножение
        x = self.v[1]*other.v[2] - self.v[2]*other.v[1]
        y = self.v[2]*other.v[0] - self.v[0]*other.v[2]
        z = self.v[0]*other.v[1] - self.v[1]*other.v[0]
        return vector(x,y,z)
    def __rmul__(self, other): #при умножении на константу, константа должна всегда стоять слева
        return vector(self.v[0]*other, self.v[1]*other, self.v[2]*other)
    def __rlshift__(self,other): # list << vector
        other += self.v
        return other
    def normalize(self):
        R = sqrt(self.v[0]**2 + self.v[1]**2 + self.v[2]**2)
        return vector(self.v[0]/R, self.v[1]/R, self.v[2]/R)
    def __str__(self):
        return "x={},y={},z={}".format(self.v[0],self.v[1],self.v[2])

class androidman():
    def __init__(self):
        self.V = []
        self.N = []
        self.T = []
        self.tfomx = 0.2
        self.tfomy = 0.1
        self.lentextrx = 0.7
        self.lentextry = 0.8

        self.coolbochka([0,0,0], [0,0,300], 200, 16, True, True) #тело
        self.coolbochka([-100,0,400], [-150,0,500], 20, 6)  #антенна
        self.coolbochka([100,0,400], [150,0,500], 20, 6)    #антенна
        self.coolbochka([240,0,250], [240,0,300], 40, 10)   #рука
        self.coolbochka([-240,0,250], [-240,0,300], 40, 10) #рука
        self.coolbochka([240,0,250], [240,0,100], 40, 10)   #рука
        self.coolbochka([-240,0,250], [-240,0,100], 40, 10) #рука
        self.coolbochka([100,0,0], [100,0,-150], 60, 10)    #нога
        self.coolbochka([-100,0,0], [-100,0,-150], 60, 10)  #нога

        self.VN = int(len(self.V)/3)
        self.TN = len(self.T)
        self.npV = np.array(self.V, dtype = np.float32).reshape(self.VN,3)
        self.npN = np.array(self.N, dtype = np.float32).reshape(self.VN,3)
        self.npT = np.array(self.T, dtype = np.float32)

        del self.V
        del self.N
        del self.T

        self.shift = np.array((0,0,0), dtype = np.float32)

    def OutVerts(self, V, N, T, Vrfom, Tfrom):
        '''перенесем свои значения векторов в глобальную систему векторов (не забудем повернуть, потом забудем!)'''

        for i in range(0, self.VN):
            V[Vrfom] = self.npV[i] + self.shift
            N[Vrfom] = self.npN[i]
            Vrfom += 1
        for i in range(0, self.TN):
            T[Tfrom] = self.npT[i]
            Tfrom += 1

        self.ifChage = False
        return (Vrfom, Tfrom)

    def RotVerts(self, xrot, zrot):
        RotM1 = np.array([ [ 1,   0, 0],
                          [0 , cos(xrot) , sin(xrot)],
                          [0, -sin(xrot) , cos(xrot)]])
        RotM2 = np.array([ [ cos(zrot),  sin(zrot), 0],
                          [-sin(zrot),  cos(zrot), 0],
                          [         0,          0, 1] ])
        RotM = np.dot(RotM1, RotM2 )
        self.npV = np.einsum('ij,kj', self.npV, RotM) #работает как вариант ниже (в кавычках), но ~x40 раз быстрее
        self.npN = np.einsum('ij,kj', self.npN, RotM)
        '''
        for i in range(0, self.VN):
            self.npV[i] = np.dot(RotM, self.npV[i])
            self.npN[i] = np.dot(RotM, self.npN[i])'''

    def coolbochka(self, v1, v2, R, N, iye = False, ifdno = False):
        ''' колбочка из точки в1 в точку в2 вот такой ширины, вот такой толщины, вот такой кривизны или с глазом '''
        alpha = 2*pi/N
        beta = alpha
        texstandart = [self.tfomx, self.tfomy, self.tfomx+self.lentextrx, self.tfomy, self.tfomx+self.lentextrx, self.tfomy+self.lentextry]
        texiye = [0, 0.05, 0, 1, 0.95, 1]
        v1 = vector(v1[0], v1[1], v1[2])
        v2 = vector(v2[0], v2[1], v2[2])
        dv = v2-v1
        NSphere = int(N/4)-1
        if NSphere < 2:
            NSphere = 2
            beta = pi/4
        e = 0.8 #типа эксцентриситет
        normmup = 1*dv
        normmup = normmup.normalize()

        if abs(dv.v[2]) < 0.00001:
            radv = (dv*vector(0,0,1)).normalize() #NormVec(VecMult(dv, (0,0,1)))
        else:
            radv = (dv*vector(0,1,0)).normalize() #NormVec(VecMult(dv, (0,1,0)))

        for i in range(0, N):
            nextradv = (dv*radv).normalize() #NormVec(VecMult(dv, radv ))
            nextradv = cos(alpha)*radv + sin(alpha)*nextradv

            self.V << R*radv + v1 << R*nextradv + v1 << R*nextradv + v2
            self.V << R*nextradv + v2 << R*radv + v2 << R*radv + v1

            self.N <<  radv << nextradv << nextradv << nextradv << radv << radv
            self.T += texstandart*2
            #и донышко закроем
            #radv = nextradv
            #continue
            if ifdno:
                self.V << R*radv + v1 << R*nextradv + v1 << v1 #+=  PlusVecs(CxV(R, radv), v1 )+ PlusVecs(CxV(R, nextradv), v1 )+ v1
                self.N << -1*normmup << -1*normmup << -1*normmup  #+= normdno*3
                self.T += texstandart
            #и круглую часть
            D0, D1 = R*radv + v2, R*nextradv + v2 #PlusVecs(CxV(R,radv), v2), PlusVecs(CxV(R,nextradv), v2)
            N0, N1 = radv, nextradv
            for j in range(0, NSphere):
                D2 = R*cos((j+1)*beta)*nextradv + e*R*sin((j+1)*beta)*normmup + v2
                D3 = R*cos((j+1)*beta)*radv     + e*R*sin((j+1)*beta)*normmup + v2
                N2 = (R*cos((j+1)*beta)*nextradv + e*R*sin((j+1)*beta)*normmup).normalize()
                N3 = (R*cos((j+1)*beta)*radv + e*R*sin((j+1)*beta)*normmup).normalize()
                self.V << D0<<D1<<D2  <<  D2<<D3<<D0 #+= D0+D1+D2  +  D2+D3+D0
                self.N << N0<<N1<<N2  <<  N2<<N3<<N0  #+= N0+N1+N2  +  N2+N3+N0
                if iye and j == 1 and i in [int(N/8), int(3*N/8)]:
                    self.T += texiye*2
                else:
                    self.T += texstandart*2
                D0, D1 = D3, D2
                N0, N1 = N3, N2
            #и пимпочка сверху
            self.V << D0 << D1 << e*R*normmup + v2
            self.N << N0 << N1 << normmup
            self.T += texstandart
            #переходим к следующей дольке
            radv = nextradv

def MatrixViev():
    #делаем матрицу вида как показано в мануале http://www.songho.ca/opengl/gl_projectionmatrix.html
    f = 5500#3000#типа дальняя граница
    r, t = int(Window.width/2), int(Window.height/2)

    n = 1500
    alpha = pi/2 #- asin(1/sqrt(2))#math.pi/6
    F = 3000#2000 #расстояние от фокуса до точвки 0 0 0

    perspective = [[n/r , 0, 0, 0],
                   [ 0, n/t, 0, 0],
                   [ 0, 0, -(f + n)/(f - n), -2*f*n/(f - n)],
                   [ 0, 0, -1, 0]]
    rotaxis     = [[ 1, 0, 0, 0],
                   [ 0, cos(alpha), sin(alpha), 0],
                   [ 0, -sin(alpha), cos(alpha), 0],
                   [ 0, 0, 0, 1]]
    shiftaxis   = [[ 1, 0, 0, 0],
                   [ 0, 1, 0, 0],
                   [ 0, 0, 1, -F],
                   [ 0, 0, 0, 1]]
    return np.dot( np.array(perspective, dtype=np.float32), np.dot( np.array(shiftaxis, dtype=np.float32), np.array(rotaxis, dtype=np.float32)))

def MatrixRot():
    alpha = pi/2 #- asin(1/sqrt(2))
    rotaxis     = [[ 1, 0, 0, 0],
                   [ 0, cos(alpha), sin(alpha), 0],
                   [ 0, -sin(alpha), cos(alpha), 0],
                   [ 0, 0, 0, 1]]
    return np.array(rotaxis, dtype=np.float32)

def GenTexture(N = 100):
    txtr = np.zeros((N*N*3), dtype = np.int8)
    for i in range(0,N):
        for j in range(0,N):
            if i > j and (i/(N-1)-0.5)**2 + (j/(N-1)-0.5)**2 < 0.25  :
                txtr[(i*N+j)*3] = 20
                txtr[(i*N+j)*3+1] = 80
                txtr[(i*N+j)*3+2] = 40
            else:
                txtr[(i*N+j)*3] = 40
                txtr[(i*N+j)*3+1] = 180
                txtr[(i*N+j)*3+2] = 75
    return txtr.tobytes()

def init():
        texID2 = glGenTextures(1)
        texID = texID2[0]

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texID)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
        glPixelStorei(GL_PACK_ALIGNMENT, 4)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 300, 300, 0, GL_RGB, GL_UNSIGNED_BYTE, GenTexture(300))

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

        glUniformMatrix4fv(  RotMatrLocation, 1 , GL_FALSE , bytes ( MatrixRot().transpose() ) )

        glEnable(GL_DEPTH_TEST)

init()

AM = androidman()

class CustomWidget(Widget):

    def __init__(self, **kwargs):
        super(CustomWidget, self).__init__(**kwargs)
        self.ReloadVertewxes = True
        self.indRender = 0

    def on_touch_move(self, touch):#on_touch_move(self, touch):
        pos = ( touch.spos[0]*2-1 , touch.spos[1]*2-1  )
        AM.RotVerts( (pos[1] - self.pos[1]), -(pos[0] - self.pos[0]) )
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

        AM.RotVerts( 0.001, 0.001 )

        self.indRender += 1
        if self.indRender%200 == 0:
            print(nap, nap**-1)

        if self.ReloadVertewxes:
            (lv, lt) = AM.OutVerts(vertices, normals, texcoords, 0, 0)
            numVertsDyrty = lv # int(lv/3)

            vertices3 = vertices.tobytes('C')
            normals3 = normals.tobytes('C')
            texcoords3 = texcoords.tobytes('C') #а вот они, кстати, не меняются при некоторой аккуратности

            glUseProgram(shaderProgram)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0,  vertices3 ) #а здесь можно попробовать на с 0 менять куски векторов
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0,  normals3 ) #.tobytes('C') )
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0,  texcoords3 )
            #self.ReloadVertewxes = False

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

        EventLoop.ensure_window()
        #мамкин хацкер режим ON
        Window.on_flip = lambda W = Window: passFunc(W)
        #мамкин хацкер режим OFF
        root = CustomWidget()

        return root

    def on_start(self):
        global numVertsDyrty
        (lv, lt) = AM.OutVerts(vertices, normals, texcoords, 0, 0)
        numVertsDyrty = lv # int(lv/3)
        print('numVertsDyrty =', numVertsDyrty)
        global vertices3
        global normals3
        global texcoords3

        vertices3 = vertices.tobytes('C')
        normals3 = normals.tobytes('C')
        texcoords3 = texcoords.tobytes('C')

        Clock.schedule_interval(self.root.update_glsl, 60 ** -1)

if __name__ == '__main__':
    MainApp().run()
