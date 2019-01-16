class Clusterer(object):

    def processLabels(self, ht):
        #helix transform equations
        ht['rt'] = np.sqrt(ht.x**2+ht.y**2)
        ht['a0'] = np.arctan2(ht.y,ht.x)
        ht['r'] = np.sqrt(ht.x**2+ht.y**2+ht.z**2)
        ht['z1'] = ht.z/ht['rt'] 
        ht['z2'] = ht.z/ht['r']
        dz0 = -0.00070
        stepdz = 0.00001
        stepeps = 0.000005
        inv = 1
        # 100 iterations of clustering
        for i in tqdm(range(100)):
            inv = inv * -1
            dz = inv*(dz0 + i*stepdz)
            ht['a1'] = ht['a0']+dz*ht['z']*np.sign(ht['z'].values)
            ht['sina1'] = np.sin(ht['a1'])
            ht['cosa1'] = np.cos(ht['a1'])
            ss = StandardScaler()
            dfs = ss.fit_transform(ht[['sina1','cosa1','z1','z2']].values)
            #scales for euclidean distance
            sc = np.array([1.0,1.0,0.4,0.4])
            for j in range(np.shape(dfs)[1]):
                dfs[:,j] *= sc[j]
            #increment eps
            clusters = DBSCAN(eps=0.0035+i*stepeps,min_samples=1,metric='euclidean',n_jobs=8).fit(dfs).labels_
            if i==0:
                ht['s1']= clusters
                ht['N1'] = ht.groupby('s1')['s1'].transform('count')
            else:
                ht['s2'] = clusters
                ht['N2'] = ht.groupby('s2')['s2'].transform('count')
                max_s1 = ht['s1'].max()
                cond = np.where(((ht['N2'].values>ht['N1'].values) & (ht['N2'].values<20)))
                s1 = ht['s1'].values
                s1[cond] = ht['s2'].values[cond]+max_s1
                ht['s1'] = s1
                ht['s1'] = ht['s1'].astype('int64')
                self.clusters = ht['s1'].values
                ht['N1'] = ht.groupby('s1')['s1'].transform('count')
        return ht['s1'].values
    
    def predict(self, hits):         
        self.clusters = self.processLabels(hits)      
        return self.clusters