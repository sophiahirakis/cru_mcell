#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# import seaborn as sns
import os
import glob
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inputdir", help="data_dir location")
args = parser.parse_args()

if not args.inputdir:
  print ('python buffers_plots -i /path/to/data_dir')
  quit()

from datetime import datetime
now = datetime.now()
runfilename = now.strftime("run_log_%m_%d_%Y_%H_%M_%S.txt")
plt.rcParams.update({'figure.max_open_warning': 0})#command for open figure warning
# Set-up default properties of plotting layout:
mpl.rcdefaults()

mpl.rcParams['figure.figsize'] = (8.375, 6.5)


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(runfilename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass  

sys.stdout = Logger()

def gen_fig(plot_data_seeds=None, plot_data_avg=None, xmax=-1, plot_title=None, plot_legend=None, plot_file=None, show_plot=False):

  fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,sharey='row')
  fig.suptitle((plot_legend + '\n' + plot_title),fontsize=14)

  for i in range(1, plot_data_seeds.shape[1]):
    ax1.plot(plot_data_seeds[:,0],plot_data_seeds[:,i])
  if xmax>-1:
    ax1.set_xlim(xmax=xmax)
  ax1.set_title('All seeds')
  ax1.set_xlabel('time (s)')
  ax1.set_ylabel('count (#)')

#  ax2.errorbar(plot_data_avg[:,0],plot_data_avg[:,1],yerr=plot_data_avg[:,2],label=plot_legend)
  ax2.plot(plot_data_avg[:,0],plot_data_avg[:,1],label=plot_legend)
  ax2.fill_between(plot_data_avg[:,0],(plot_data_avg[:,1]-plot_data_avg[:,2]),(plot_data_avg[:,1]+plot_data_avg[:,2]),color='gray',alpha=0.2)
  if xmax>-1:
    ax2.set_xlim(xmax=xmax)
  ax2.set_title('Mean & Stddev')
  ax2.set_xlabel(r'time (s)')

  if plot_file:
    plt.savefig(plot_file + '.pdf', dpi=600)
#    plt.savefig(plot_file + '.png', dpi=300)

  if (show_plot):
    plt.show()



# data_dir = '/cnl/mcelldata/Sophie/2021/Saftenku_RyR-no_cytClamp-h-LTCC_9_blend'
# data_dir = '/home/avwave/store/repos/notebooks/Saftenku_RyR-no_cytClamp-h-LTCC_10_blend'
data_dir = args.inputdir
cases = [
              ['saftenku_ryr_spark_SRclamp_CaC_fet_0.01_files','saftenku RyR Spark CaC fet 0.1','sRyRCaC5rfet01'],
              ['saftenku_ryr_spark_SRclamp_CaC_fet_0.05_files','saftenku RyR Spark CaC fet 0.05','sRyRCaC5rfet005'],
              ['saftenku_ryr_spark_SRclamp_CaC_fet_0.1_files','saftenku RyR Spark CaC fet 0.1','sRyRCaC5rfet01'],
              ['saftenku_ryr_spark_SRclamp_CaC_fet_1.0_files','saftenku RyR Spark CaC fet 1.0','sRyRCaC5rfet10'],
        ]
react_suffix= 'mcell/output_data/react_data'
plot_suffix= 'plots'

seed_glob = 'seed_*'
#   [data_file_name, data label, plot_file_prefix, xmax_value (in seconds where -1 means no limit))

react_data_files = [
                     ['Ca2_sr.World.dat', 'SR Calcium', 'SRCalcium', -1 ],
                     ['Ca2_cyt.World.dat', 'Cytosolic Calcium', 'CytCalcium', -1 ],
                     ['PMCA0.World.dat','Plasma membrane Calcium Pump, State 0', 'pmca0', -1],
                     ['PMCA1.World.dat','Plasma membrane Calcium Pump, State 1', 'pmca1',  -1],
                     ['NCX0.World.dat','Sodium-Calcium Exchanger, State 0', 'ncx0', -1],
                     ['NCX1.World.dat','Sodium-Calcium Exchanger, State 1', 'ncx1', -1],
                     ['CMDN_cyt.World.dat','Calmodulin','cmdn', -1],
                     ['CSQN_sr.World.dat','Calsequestrin','csqn', -1],
                     ['TRPN.World.dat','Troponin C','trpn', -1],
                     ['CMDN_Ca.World.dat','Calcium-Bound Calmodulin','cmdnCalcium',-1],
                     ['CSQN_Ca.World.dat','Calcium-Bound Calsequestrin','csqnCalcium',-1],
                     ['TRPN_Ca.World.dat','Calcium-Bound Troponin C','trpnCalcium', -1],
                     ['ATP_Ca.World.dat', 'Calcium-Bound ATP', 'atpCalcium', -1 ],
                     ['FLO5_sr.World.dat','Fluo-5','flo5', -1],
                     ['FLO4_cyt.World.dat','Fluo-4','flo4', -1],
                     ['FLO5_Ca.World.dat','Calcium-Bound Fluo-5','flo5Calcium', -1],
                     ['FLO4_Ca.World.dat','Calcium-Bound Fluo-4','flo4Calcium', -1],
                     ['Ca2_cyt.jDyad_sample_volume.dat','Calcium in dyadic junction','dyadCalcium', -1],
                     ['Ca2_sr.jSR_sample_volume.dat','Calcium in junctional SR','jsrCalcium', -1],
                     ['FLO4_cyt.jDyad_sample_volume.dat','Fluo-4 in junctional SR','jsrFlo4', -1],
                     ['FLO4_Ca.jDyad_sample_volume.dat','Calcium-Bound Fluo-4 in junctional SR','jsrFlo4Ca', -1],
                     ['FLO5_sr.jSR_sample_volume.dat','Fluo-5 in junctional SR','jsrFlo5', -1],
                     ['FLO5_Ca.jSR_sample_volume.dat','Calcium-Bound Fluo-5 in junctional SR','jsrFlo5Ca', -1],
                   ]
errorFiles = []

for case in cases:
    output_dir = data_dir + '/' + case[0] + '/' + react_suffix
    plot_data_dir = data_dir + '/' + plot_suffix
    data_glob = data_dir + '/' + case[0] + '/' + react_suffix + '/' + seed_glob 
    print('data_glob: ' + data_glob)
    for react_data_file in react_data_files:
      react_data_file_glob = data_glob + '/' + react_data_file[0]
      all_react_data_files = sorted(glob.glob(react_data_file_glob))

      data = []
      for dat_file in all_react_data_files:
        print('')
        if len(data) == 0:
          print('Initializing data: %s' % (dat_file))
          data = np.fromfile(dat_file, sep=' ')   # read data into a single row: x0,y0,x1,y1,x2,y2...
          data = data.reshape((-1,2))  # reshape data into 2 columns [x0, y0]...
        else:
          print('Accumulating data: %s' % (dat_file))
          try:
            d = np.fromfile(dat_file, sep=' ')  # read and reshape next data file as above
            d = d.reshape((-1,2))
            data = np.append( data, d[:,1].reshape(-1,1), 1 )  # append second column of d to data as a new column
          except ValueError as val_error:
            print('WARNING in %s check number of rows' % dat_file)

      if (np.array(data).size > 0) : #ensure data contains something before proceeding line below does not like empty arrays
        avg = data[:,1:].mean(axis=1).reshape(-1,1)
        stddev = data[:,1:].std(axis=1).reshape(-1,1)
        outdata = data[:,0].reshape(-1,1)
        outdata = np.append( outdata, avg, 1 )
        outdata = np.append( outdata, stddev, 1 )

        fn_out = os.path.splitext(react_data_file[0])[0]+ '.avg_stddev.dat'
        fn_out = os.path.join(output_dir,fn_out)
        print('')
        print('Writing avg & stddev data: %s' % (fn_out))
        print('')
        np.savetxt(fn_out,outdata,fmt= '%.14g',delimiter=' ',newline='\n')

        #gen_fig(plot_data_seeds=data, plot_data_avg=outdata, xmax=react_data_file[3],  plot_title=case[1], plot_legend=react_data_file[1], plot_file=os.path.join(plot_data_dir, react_data_file[2] + '_' + case[2]), show_plot=False)

        #plt.close('all') 
