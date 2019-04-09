# This python script contains codes for channel generation, lifting 
# and precoder prediction

from __future__ import division
import itpp
import numpy as np
import scipy.linalg as la
import pdb
np.random.seed(1)

def sH_lift(A,B,ret_vec=False):
    p=A.shape[1]
    n=A.shape[0]
    X=np.matrix(A)
    Y=np.matrix(B)
    Xu=X[0:p]
    Xl=X[p:n]
    Yu=Y[0:p]
    Yl=Y[p:n]
    C=2*la.inv((Xu.H+Yu.H))*skew(Yu.H*Xu+Xl.H*Yl)*la.inv(Xu+Yu)
    B=(Yl-Xl)*la.inv(Xu+Yu)
    T=np.bmat([[C,-B.H],[B,np.zeros((n-p,n-p))]])
    cvecC=np.array(C[np.triu_indices(C.shape[0],1)]).flatten()
    rvecC=np.append(np.imag(np.diagonal(C)),np.append(np.imag(cvecC),np.real(cvecC)))
    cvecB=np.squeeze(np.asarray(np.reshape(B,(1,(n-p)*p))))
    rvecB=np.append(np.imag(cvecB),np.real(cvecB))
    vecT=np.append(rvecC,rvecB)
    if(ret_vec):
        return T,vecT
    else:
        return T

def vec_to_tangent(vec,n,p):
    C=np.diag(1j*vec[:p])
    C[np.triu_indices(C.shape[0],1)]=1j*vec[p:p+(p*(p-1)/2)]+vec[p+(p*(p-1)/2):p+p*(p-1)]
    C[np.tril_indices(C.shape[0],-1)]=1j*vec[p:p+(p*(p-1)/2)]-vec[p+(p*(p-1)/2):p+p*(p-1)]
    vec_recon=1j*vec[p+p*(p-1):p+p*(p-1)+p*(n-p)]+vec[p+p*(p-1)+p*(n-p):p+p*(p-1)+2*p*(n-p)]
    B=np.matrix(np.reshape(vec_recon,(n-p,p)))
    T=np.bmat([[C,-B.H],[B,np.zeros((n-p,n-p))]])
    return T

def sH_retract(A,B):
    p=A.shape[1]
    n=A.shape[0]
    X=np.matrix(A)
    W=np.matrix(B)
    Cay_W=(np.identity(n)+W)*la.inv(np.identity(n)-W)
    un_normQT= Cay_W*X
    norm_Qt= un_normQT/la.norm(un_normQT,axis=0)
    return norm_Qt

def skew(A):
    A_mat=np.matrix(A)
    return 0.5*(A.H-A)

def Ds_metric(A,B):
    A_mat=np.matrix(A)
    B_mat=np.matrix(B)
    return A.shape[1]-np.trace(np.square(np.abs(A_mat.H*B_mat)))

def chordal_dist(A,B):
    A_mat=np.matrix(A)
    B_mat=np.matrix(B)
    return np.sqrt(np.abs(A.shape[1]-np.linalg.norm(A_mat.H*B_mat,'fro')**2))

def grassCD_2(A,B):
    C=np.vdot(A,B)
    rho=np.real(C*np.conjugate(C))
    return (1-rho)

def stiefCD(A,B):
    A_mat=np.matrix(A)
    B_mat=np.matrix(B)
    CD_2=np.sum([grassCD_2(np.array(A_mat[:,i]),np.array(B_mat[:,i])) for i in range(A_mat.shape[1])])
    if(CD_2<0):
        pdb.set_trace()
        return -1
    return np.sqrt(CD_2)


def qtisn(pU,rU,g,num_iter,sH_list,norm_fn,sk=0.0):
    np.random.seed(1)
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
            # print("Yes")
            chordal_dists=np.array([stiefCD(Q[i],rU) for i in range(len(Q))])

        trials=trials+1
        if(np.argmin(chordal_dists)<(len(Q)//2)):
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
    for i in range(n):
        U[:,i]=U[:,i]/la.norm(U[:,i])
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


def quasiGeodinterp(qtized_Us, num_subcarriers, feedback_indices):
    recon_U=np.zeros((num_subcarriers,qtized_Us.shape[1],qtized_Us.shape[2]),dtype='complex')
    recon_U[feedback_indices]=qtized_Us
    separation=int((num_subcarriers-1)/(qtized_Us.shape[0]-1))
    for i in range(qtized_Us.shape[0]-1):    
        U_current=qtized_Us[i]
        U_next=qtized_Us[i+1]
        M=np.matrix(U_next)
        S=sH_lift(U_current,M)
        t=np.arange(1,separation)/separation
        interpolation_fn= lambda  t:sH_retract(U_current,t*S)
        U_interpolate=list(map(interpolation_fn, t))
        recon_U[feedback_indices[i]+1:feedback_indices[i+1]]=U_interpolate
    return recon_U

def sigmaInterp_qtize(sigma_data,num_subcarriers,feedback_indices,sigma_cb):
    interpS=np.zeros((num_subcarriers,sigma_data.shape[1]))
    diff_freq=feedback_indices[1]-feedback_indices[0]
    for i in range(feedback_indices.shape[0]-1):
        curr_S = sigma_data[i]
        next_S = sigma_data[i+1]
        qcurr_S=sigma_cb[np.argmin([la.norm(curr_S-codeword) for codeword in sigma_cb])]
        qnext_S=sigma_cb[np.argmin([la.norm(next_S-codeword) for codeword in sigma_cb])]
        interpS[feedback_indices[i]:feedback_indices[i+1]]=[(1-(t/diff_freq))*qcurr_S+(t/diff_freq)*qnext_S\
            for t in range(0,diff_freq)]
    interpS[feedback_indices[feedback_indices.shape[0]-1]]=qnext_S
    return interpS

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

def onlyT_pred(center,pred_list):
    np.random.seed(1)
    num_iter=20
    i=0
    Np=pred_list.shape[0]
    while (i<num_iter):
        tangent_list=np.array([lift(center,manifold_pt) for manifold_pt in pred_list])
        new_tangent=np.sum(tangent_list,axis=0)/Np
        center=retract(center,new_tangent)
        i+=1
    tangent_list=np.array([lift(center,manifold_pt) for manifold_pt in pred_list])
    sum_itp=np.sum(np.array([(i+1)*tangent_list[i] for i in range(Np)]),axis=0)
    sum_tp=np.sum(tangent_list,axis=0)
    T_1=((-sum_itp+((1+Np)/2)*sum_tp)*12)/((Np-1)*(Np**2+Np))
    T_0=(sum_tp-((Np-1)*Np/2)*T_1)/Np
    pU=retract(center,T_0+Np*T_1)
    return pU


def lift(A,B):
    p=A.shape[0]
    n=A.shape[1]
    A_mat=np.matrix(A)
    B_mat=np.matrix(B)
    T=(np.identity(p)-A_mat*A_mat.H)*B_mat+0.5*A_mat*(A_mat.H*B_mat-B_mat.H*A_mat)
    return T

def retract(A,V):
    p=A.shape[0]
    n=A.shape[1]
    A_mat=np.matrix(A)
    V_mat=np.matrix(V)
    S=(A_mat+V_mat)*la.sqrtm(la.inv((np.identity(n)+V_mat.H*V)))
    return S


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