# This python script contains codes for channel generation, lifting 
# and precoder prediction

from __future__ import division
import itpp
import numpy as np
import scipy.linalg as la
import pdb

def qtisn(pU,rU,g,num_iter,sH_list,norm_fn,sk=0.0):
    diff_frob_norm = lambda A,B:np.linalg.norm(A-B, 'fro')
    trials=0
    sp=0.0
    sm=0.0
    while(trials<num_iter):
        sp=g**(np.min(sk+1,0))
        sm=g**(sk-1)
        # Qp=[np.matmul(la.expm(sp*sH),pU) for sH in sH_list]
        # Qm=[np.matmul(la.expm(sm*sH),pU) for sH in sH_list]
        Qp=[sH_retract(pU,sp*sH) for sH in sH_list]
        Qm=[sH_retract(pU,sm*sH) for sH in sH_list]
        Q=Qp+Qm
        if(norm_fn=="diff_frob_norm"):
            chordal_dists=np.array([diff_frob_norm(Q[i],rU) for i in range(len(Q))])
        else:
            chordal_dists=np.array([stiefCD(Q[i],rU) for i in range(len(Q))])

        trials=trials+1
        if(np.argmin(chordal_dists)<len(Q)//2):
            sk=np.min(sk+1,0)
        else:
            sk=sk-1
    #print("Qtisn error: "+str(np.min(chordal_dists)))
    return np.min(chordal_dists),Q[np.argmin(chordal_dists)]

def vec_to_unitary(vec,n):
    U=np.zeros((n,n),dtype=complex)
    prev=0
    nex=2*n
    i=0
    while(i!=n):
        U[:,i][0:n-i]=vec[prev:nex-1][::2]+1j*vec[prev+1:nex][::2]
        if(i>0):
            inn_pros=-1*np.array([np.vdot(U[:,i][0:n-i],U[:,j][0:n-i]) for j in range(0,i)])
            submat=np.matrix(U[np.ix_(np.arange(n-i,n),np.arange(0,i))]).T
            # pdb.set_trace()
            U[:,i][n-i:n]=np.conjugate(np.matmul(la.inv(submat),inn_pros))
        prev=nex
        nex+=2*n-2*(i+1)
        i+=1
    return U

def unitary_to_vec(U):
    n=U.shape[0]
    vec=np.zeros(n**2+n)
    prev=0
    nex=2*n
    i=0
    while(i!=n):
        vec[prev::2][0:n-i]=np.array(U[:,i][0:n-i]).flatten().real
        vec[prev+1::2][0:n-i]=np.array(U[:,i][0:n-i]).flatten().imag
        prev=nex
        nex+=2*n-2*(i+1)
        i+=1
    return vec

def vec_to_semiunitary(vec,m,n):
    U=np.zeros((m,n),dtype=complex)
    prev=0
    nex=2*m
    i=0
    while(i!=n):
        U[:,i][0:m-i]=vec[prev:nex-1][::2]+1j*vec[prev+1:nex][::2]
        if(i>0):
            inn_pros=-1*np.array([np.vdot(U[:,i][0:m-i],U[:,j][0:m-i]) for j in range(0,i)])
            submat=np.matrix(U[np.ix_(np.arange(m-i,m),np.arange(0,i))]).T
            # pdb.set_trace()
            U[:,i][m-i:m]=np.conjugate(np.matmul(la.inv(submat),inn_pros))
        prev=nex
        nex+=2*m-2*(i+1)
        i+=1
    return U

def semiunitary_to_vec(U):
    m=U.shape[0]
    n=U.shape[1]
    vec=np.zeros(2*m*n-n*(n-1))
    prev=0
    nex=2*m
    i=0
    while(i!=n):
        vec[prev::2][0:m-i]=np.array(U[:,i][0:m-i]).flatten().real
        vec[prev+1::2][0:m-i]=np.array(U[:,i][0:m-i]).flatten().imag
        prev=nex
        nex+=2*m-2*(i+1)
        i+=1
    return vec

def unitary(n):
    X=(np.random.rand(n,n)+1j*np.random.rand(n,n))/np.sqrt(2)
    [Q,R]=np.linalg.qr(X)
    T=np.diag(np.diag(R)/np.abs(np.diag(R)))
    U=np.matrix(np.matmul(Q,T))
    # Verify print (np.matmul(U,U.H))
    return U    
    
def rand_stiefel(n,p):
    H=(np.random.rand(n,p)+1j*np.random.rand(n,p))/np.sqrt(2)
    U, S, V = np.linalg.svd(H,full_matrices=0)
    return U

def rand_SH(n):
    X=2*np.random.rand(n,n)-np.ones((n,n))+1j*(2*np.random.rand(n,n)-np.ones((n,n)))
    A=np.matrix(X-X.T.conjugate())/2
    # Verify print(A-A.H)
    return A


def find_precoder_list(H_list,ret_full=False):
    V_list = []
    U_list = []
    sigma_list=[]
    if(ret_full):
        fV_list=[]
        fU_list=[]
        fsigma_list=[]
    for H_matrix in H_list:
        H_matrix=np.array(H_matrix)
        U, S, V = np.linalg.svd(H_matrix,full_matrices=0)
        V = np.transpose(np.conjugate(V))
        V_list.append(np.matrix(V))
        newU=np.matmul(U,np.diag(np.exp(-1j*np.angle(U[0,:]))))
        U_list.append(np.matrix(newU))
        sigma_list.append(S)
        if(ret_full):
            U, S, V = np.linalg.svd(H_matrix,full_matrices=1)
            V = np.transpose(np.conjugate(V))
            newU=np.matmul(U,np.diag(np.exp(-1j*np.angle(U[0,:]))))
            fU_list.append(np.matrix(newU))
            fsigma_list.append(S)
            fV_list.append(V)
    if(ret_full):
        return np.array(V_list),np.array(U_list),np.array(sigma_list),np.array(fU_list),np.array(fsigma_list),np.array(fV_list)
    else:
        return np.array(V_list),np.array(U_list),np.array(sigma_list)

def apply_interpolation(self, feedback_list, Nt, Nr,interpType,qtCodebook=None,retQTnoise=False):
    diff_frob_norm = lambda A,B:np.linalg.norm(A-B, 'fro')
    feedback_indices=self.fb_indices
    interpolated_U_list=np.zeros((self.num_subcarriers, Nt, Nr),dtype=complex)
    qtNoise=np.zeros(feedback_indices.shape[0])
    for i in range(self.num_fb_points-1):
        if(qtCodebook is not None):
            thU_curr=feedback_list[feedback_indices[i]]
            thU_next=feedback_list[feedback_indices[i+1]]
            U_current=qtCodebook[np.argmin([diff_frob_norm(thU_curr,codeword)\
                for codeword in qtCodebook])]
            U_next=qtCodebook[np.argmin([diff_frob_norm(thU_next,codeword)\
             for codeword in qtCodebook])]
            qtNoise[i]+=diff_frob_norm(thU_curr,U_current)
            if(i==self.num_fb_points-2):
                qtNoise[i+1]+=diff_frob_norm(thU_next,U_next)
        else:            
            U_current=feedback_list[feedback_indices[i]]
            U_next=feedback_list[feedback_indices[i+1]]
        num_indices_to_be_filled=feedback_indices[i+1]-feedback_indices[i]-1
        if(i==self.num_fb_points-2):
            if(interpType=='Geodesic'):
                interpolated_U_list[feedback_indices[i]:feedback_indices[i+1]+1]=\
                self.Geodesic_interpolation(U_current, U_next, num_indices_to_be_filled,Nr, last_fill_flag=True)
            if(interpType=='quasiGeodesic'):
                interpolated_U_list[feedback_indices[i]:feedback_indices[i+1]+1]=\
                self.quasiGeodesic_interpolation(U_current, U_next, num_indices_to_be_filled, last_fill_flag=True)
            if(interpType=='orthLifting'):
                interpolated_U_list[feedback_indices[i]:feedback_indices[i+1]+1]=\
                self.orthLifting_interpolation(U_current, U_next, num_indices_to_be_filled, last_fill_flag=True)
        else:
            if(interpType=='Geodesic'):
                interpolated_U_list[feedback_indices[i]:feedback_indices[i+1]]=\
                self.Geodesic_interpolation(U_current, U_next, num_indices_to_be_filled,Nr, last_fill_flag=False)
            if(interpType=='quasiGeodesic'):
                interpolated_U_list[feedback_indices[i]:feedback_indices[i+1]]=\
                self.quasiGeodesic_interpolation(U_current, U_next, num_indices_to_be_filled, last_fill_flag=False)
            if(interpType=='orthLifting'):
                interpolated_U_list[feedback_indices[i]:feedback_indices[i+1]]=\
                self.orthLifting_interpolation(U_current, U_next, num_indices_to_be_filled, last_fill_flag=False)                    
    if(retQTnoise):
        return interpolated_U_list,qtNoise
    else:
        return interpolated_U_list

def Geodesic_interpolation(self, U_current, U_next, num_indices_to_be_filled, Nr, last_fill_flag=False):
    orientation_matrix=self.find_orientation_matrix(U_current, U_next)
    M=np.matrix(la.inv(U_current))*np.matrix(U_next)*np.matrix(orientation_matrix)
    S=la.logm(M)
    t=np.linspace(0,1,num=num_indices_to_be_filled+1, endpoint=False)
    interpolation_fn= lambda  t:(np.matrix(U_current)*np.matrix(la.expm(t*S)))[:,np.arange(Nr)]
    U_interpolate=list(map(interpolation_fn, t))
    if(last_fill_flag):
        U_interpolate.append(U_next[:,np.arange(Nr)])
    return np.array(U_interpolate)

def quasiGeodesic_interpolation(self, U_current, U_next, num_indices_to_be_filled, last_fill_flag=False):
    # orientation_matrix=self.find_orientation_matrix(U_current, U_next)
    M=np.matrix(U_next)
    S=sH_lift(U_current,M)
    t=np.linspace(0,1,num=num_indices_to_be_filled+1, endpoint=False)
    interpolation_fn= lambda  t:sH_retract(U_current,t*S)
    U_interpolate=list(map(interpolation_fn, t))
    if(last_fill_flag):
        U_interpolate.append(U_next)
    return np.array(U_interpolate)

def orthLifting_interpolation(self, U_current, U_next, num_indices_to_be_filled, last_fill_flag=False):
    orientation_matrix=self.find_orientation_matrix(U_current, U_next)
    M=np.matrix(U_next)*np.matrix(orientation_matrix)
    S=lift(U_current,M)
    t=np.linspace(0,1,num=num_indices_to_be_filled+1, endpoint=False)
    interpolation_fn= lambda  t:retract(U_current,t*S)
    U_interpolate=list(map(interpolation_fn, t))
    if(last_fill_flag):
        U_interpolate.append(U_next)
    return np.array(U_interpolate)

def find_orientation_matrix(self, U0, U1):
    A_current= np.diag(np.matmul(U1.T.conj(),U0))
    orientaion_matrix=np.diag((A_current/np.absolute(A_current)))
    return orientaion_matrix

def convex_interpolation(self, S_current, S_next, num_indices_to_be_filled, last_fill_flag=False):
    t=np.linspace(0,1,num=num_indices_to_be_filled+1, endpoint=False)
    interpolation_fn= lambda  t:t*S_current+(1-t)*S_next
    S_interpolate=list(map(interpolation_fn,t))
    if(last_fill_flag):
        S_interpolate.append(S_next)
    return np.array(S_interpolate)

# Class for generating a MIMO TDL channel
class MIMO_TDL_Channel():
    # Class Constructor
    def __init__(self,Nt,Nr,c_spec,sampling_time,Num_subcarriers):
        # Num Tx antenna
        self.Nt=Nt
        # Num Rx antenna
        self.Nr=Nr
        # Sampling Time
        self.sampling_time = sampling_time
        # FFT size will be the number of subcarriers in OFDM frame
        self.fft_size = Num_subcarriers
        # Declare Nt*Nr channels        
        self.tdl_channels=[]
        # Initialise using the TDL Channel method to get the channel length
        # Delay profile in cspec will be discretized with sampling_time in seconds
        channel = itpp.comm.TDL_Channel(c_spec, sampling_time)
        self.channel_length=channel.taps()
        # Initialise the Nt*Nr TDL channels
        for i in range(self.Nt*self.Nr):
            self.tdl_channels.append(itpp.comm.TDL_Channel(c_spec,sampling_time))

    # Set Doppler for each Nt*Nr channel
    def set_norm_doppler(self,norm_doppler):
        for i in range(self.Nt*self.Nr):
            self.tdl_channels[i].set_norm_doppler(norm_doppler)

    # Generate method to 
    def generate(self):
        # Declare a CMAT channel matrix of Nt*Nr size
        self.channel=itpp.cmat()
        self.channel.set_size(self.Nt*self.Nr,self.channel_length,False)
        # Declare another temp CMAT
        channel_coef_one=itpp.cmat()
        # Initialise the matrix element wise
        for i in range(self.Nt):
            for j in range(self.Nr):
                # Generate "1" sample values of the channel,  
                # channel_coef_one has one tap per column                 
                self.tdl_channels[i*self.Nr+j].generate(1,channel_coef_one)
                for l in range(self.channel_length):
                    self.channel.set(i*self.Nr+j,l,channel_coef_one(0,l))

    # Function to get Precoder List
    def get_Hlist(self):
        # Np array to store `FFT_size` number of Nt*Nr channels
        chan_freq=np.zeros(shape=(self.fft_size,self.Nt,self.Nr),dtype=complex)
        # Initialise the matrices        
        for i in range(self.Nt*self.Nr):
            col_idx=i%self.Nr
            row_idx=i//self.Nr
            freq_resp=itpp.cmat()
            inp_resp=itpp.cmat()
            #Get Impulse Response
            inp_resp.set_size(1,self.channel_length,False)
            inp_resp.set_row(0,self.channel.get_row(i))
            # print(inp_resp)
            #Calculate Frequency Response for each Nt*Nr channel (64 length array)
            self.tdl_channels[i].calc_frequency_response(inp_resp,freq_resp , 2*self.fft_size)
            # Store it in chan_freq in appropriate places across 64 matrices
            # print(freq_resp.to_numpy_ndarray().flatten()[0:64]-freq_resp.to_numpy_ndarray().flatten()[64:128])
            chan_freq[:,row_idx][:,col_idx]=freq_resp.to_numpy_ndarray().flatten()[0:self.fft_size]
        
        chan_freq_list=[]
        for i in range(self.fft_size):
            chan_freq_list.append(chan_freq[i])
        #print(chan_freq_list[0])
        return(chan_freq_list)