# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 13:46:56 2020

@author: aminebennani
"""

# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

"""

# import the different functionnalities necessary for work

import numpy as np
import matplotlib.pyplot as plt

from mne import Epochs, pick_types, events_from_annotations, set_log_level
from mne.preprocessing import ICA
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.channels import read_layout

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score


class eeg:
    
    
    def load_data(self,subject):
        """
        Load data from a subject designated by his identifier.
        Input:
        subject: int between 1 and 109
        Output:
        raw: mne.Raw, structure containing EEG data
        events: numpy array (n_events, 3)
        first column: date of event in sample
        second column: duration of event
        thrid column: event code
        """
        assert 1 <= subject <= 109
        
        # dictionnary to specify the label and code of each event of interest
        event_id = dict(left=0, right= 1, hands=2, feet=3)
        
        # list of dictionnaries to specify the different tasks of interest
        task = [
                dict(T1=event_id['left'], T2=event_id['right']),
                dict(T1=event_id['hands'], T2=event_id['feet'])
        ]
        
        # list of dictionnaries to specify the different runs to load for one subject
        runs = [
            dict(id=4, task=task[0]),
            dict(id=6, task=task[1]),
            dict(id=8, task=task[0]),
            dict(id=10, task=task[1]),
            dict(id=12, task=task[0]),
            dict(id=14, task=task[1])
        ]
        
        # load and concatenate the different files from the specified subject
        # download the files if necessary
        raws = list()
        events_list = list()
        for run in runs:
            # localize the file, download it if necessary
            filename = eegbci.load_data(subject, run['id'])
            # load its contain
            raw = read_raw_edf(filename[0], preload=True)
            events, _ = events_from_annotations(raw, event_id=run['task'])
            # accumulate the data
            raws.append(raw)
            events_list.append(events)
            
        # concatenate all data in two structures : one for EEG, one for the events
        raw, events = concatenate_raws(raws, events_list=events_list)
        
        # strip channel names of "." characters
        raw.rename_channels(lambda x: x.strip('.'))
        
        # delete annotations
        indices = [x for x in range(len(raw.annotations))]
        indices.reverse()
        
        for i in indices:
            raw.annotations.delete(i)
        
        return raw, events
    
    
    
    def model(self, n_components):
        """
        Classification using Linear Discriminant Analysis (lda)
        Signal decomposition using Common Spatial Patterns (CSP)
        """
        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
        clf = Pipeline([('CSP', csp), ('LDA', lda)])
        
        return clf, csp
   
    
    def evaluate(self, clf, data_train, cv, labels, n_jobs):
        """
        Evaluate the score using cross-validation 
        """
        return cross_val_score(clf, data_train, labels, cv=cv, n_jobs=n_jobs)
    
    
    # Drop artefacts using ICA method
    def drop_artefacts(self, n_components, raw, a, b):
        """
        Use the ICA to eliminate windows contaminated with blinking
        """
        ica = ICA(n_components=n_components)
        ica.fit(raw)
        picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                           exclude='bads')
        raw.load_data()
        ica.plot_sources(raw)
        ica.plot_components(layout=read_layout('eeg1005'))
            
        ica.exclude = [a,b]
        reconst_raw = raw.copy()
        ica.apply(reconst_raw)
        
        return reconst_raw
    
    
    def main(self):
        
        raw1,events1 = self.load_data(1)
        
        event_dict = {'Main droite': 1,'Deux pieds': 3}
        raw = []
        events = []
        epochs = []
        rawf = []
        tmin, tmax = -1., 4.
        scoresmean = []
        
        set_log_level(verbose='CRITICAL')
        
        for k in range(8):
            raws,eventss = self.load_data(k+1)
            raw.append(raws)
            events.append(eventss)
            
            rawf.append(raw[k].copy().filter(8,30,fir_design='firwin'))
            picks = pick_types(rawf[k].info, meg=False, eeg=True, stim=False, eog=False,
                           exclude='bads')
            
            epochs.append(Epochs(rawf[k], events[k], event_dict, tmin, tmax, proj=True, picks=picks,
                        baseline=None, preload=True))
            
            labels = epochs[k].events[:,-1]
            
            scores = []
            epochs_data_train = epochs[k].get_data()
            cv = ShuffleSplit(10, test_size=0.2, random_state=42)
            
            #cv_split = cv.split(epochs_data_train) 
            
            # Create the classifier model
            clf,csp = self.model(4)
            
            # Model Evaluation
            scores = self.evaluate(clf, epochs_data_train, cv, labels, 1)
            
            print("Subject %i, Classification accuracy: %f" %(k+1,np.mean(scores)))
            scoresmean.append(np.mean(scores))
            
            # CSP patterns estimated on full data
            csp.fit_transform(epochs_data_train, labels)
           
            if k== 2 or k == 6 or k==19 or k==7:   
                layout = read_layout('eeg1005')
                csp.plot_patterns(epochs[k].info, layout=layout, ch_type='eeg',
                                      units='patterns (au)', size=1.5)
                
                plt.title("Subject %i, Classification accuracy: %f" %(k+1,np.mean(scores)))
                plt.show()
        
        # Eliminate artefacts
        reconst_raw = self.drop_artefacts(10, raw1, 0, 2)       
        
        raw1.plot()
        reconst_raw.plot()



if __name__=="__main__":
    e = eeg()
    e.main()
    
