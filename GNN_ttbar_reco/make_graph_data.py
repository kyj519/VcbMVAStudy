import os, sys
import numpy as np
import tqdm, argparse, htcondor
import multiprocessing
import itertools
import ROOT


def RvecToNumpy(original_arr,key,item):
    
    temp = []
    for i in tqdm.tqdm(range(item.shape[0])):
       temp.append(np.asarray(item[i]))
    original_arr[key] = np.asarray(temp)
    
def calculate_eta(pt, pz):
    theta = np.arctan2(pt, pz)
    eta = -np.log(np.tan(theta/2))
    return eta

def calculate_deltaR(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = phi1 - phi2
    # Normalize dphi to be between -pi and pi
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    deltaR = np.sqrt(deta**2 + dphi**2)
    return deltaR


def calculate_mass(E1, pt1, eta1, phi1, E2, pt2, eta2, phi2):
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)

    total_px = px1 + px2
    total_py = py1 + py2
    total_pz = pz1 + pz2
    total_E = E1 + E2

    total_mass_squared = total_E**2 - total_px**2 - total_py**2 - total_pz**2
    total_mass = np.sqrt(total_mass_squared)

    return total_mass

ROOT.EnableImplicitMT(16)
def makeGraph(file,out_path):
    df = ROOT.RDataFrame('GNN_tree',file)
    arr = df.AsNumpy()
    arr['met_E'] = np.sqrt(arr['met_pt']**2+arr['met_pz']**2)

    arr['met_eta'] = calculate_eta(arr['met_pt'],arr['met_pz'])

    graph_arr = []

    for i in tqdm.tqdm(range(arr['weight'].shape[0])):
        ## node construction
        
        lepton_node = np.zeros((1,3))
        lepton_node[0,0] = arr['lepton_E'][i]
        lepton_node[0,1] = arr['lepton_pt'][i]
        lepton_node[0,2] = arr['lepton_charge'][i]
        
        met_node = np.zeros((1,3))
        met_node[0,0] = arr['met_E'][i]
        met_node[0,1] = arr['met_pt'][i]
        met_node[0,2] = 0
        
        
        jet_E = np.asarray(arr['vec_sel_jet_match_E'][i])
        jet_pt = np.asarray(arr['vec_sel_jet_match_pt'][i])
        jet_phi = np.asarray(arr['vec_sel_jet_match_phi'][i])
        jet_eta = np.asarray(arr['vec_sel_jet_match_eta'][i])
        
        jet_node = np.concatenate((jet_E,jet_pt,
                                #np.asarray(arr['vec_sel_jet_match_charge'][i]),
                                np.asarray(arr['vec_sel_jet_match_bvsc'][i]),
                                np.asarray(arr['vec_sel_jet_match_cvsb'][i]),
                                np.asarray(arr['vec_sel_jet_match_cvsl'][i])))
        jet_node = np.reshape(jet_node,(jet_E.shape[0],5))
        total_node_N = 2 + jet_node.shape[1]
        edge_idx = []
        edge_attr=[]
        edge_y=[]
        
        ##edge between lepton and others
        
        edge_idx.append([0, 1])
        dR = calculate_deltaR(arr['lepton_eta'][i],arr['lepton_phi'][i],arr['met_eta'][i],arr['met_phi'][i])
        dimass = calculate_mass(arr['lepton_E'][i],arr['lepton_pt'][i],arr['lepton_eta'][i],arr['lepton_phi'][i],
                                arr['met_E'][i],arr['met_pt'][i],arr['met_eta'][i],arr['met_phi'][i])
        
        edge_attr.append([dR,dimass])
        edge_y.append(1)
        
        for jet_idx in range(jet_node.shape[0]):
            edge_idx.append([0, jet_idx])
            dR = calculate_deltaR(arr['lepton_eta'][i],arr['lepton_phi'][i],jet_eta[jet_idx],jet_phi[jet_idx])
            dimass = calculate_mass(arr['lepton_E'][i],arr['lepton_pt'][i],arr['lepton_eta'][i],arr['lepton_phi'][i],
                                    jet_E[jet_idx],jet_pt[jet_idx],jet_eta[jet_idx],jet_phi[jet_idx])
            edge_attr.append([dR,dimass])
            if arr['idx_jet_lep_t_b'][i] == jet_idx:
                edge_y.append(1)
            else:
                edge_y.append(0)
                
        ##edge between met and others
        
        for jet_idx in range(jet_node.shape[0]):
            edge_idx.append([1, jet_idx])
            dR = calculate_deltaR(arr['met_eta'][i],arr['met_phi'][i],jet_eta[jet_idx],jet_phi[jet_idx])
            dimass = calculate_mass(arr['met_E'][i],arr['met_pt'][i],arr['met_eta'][i],arr['met_phi'][i],
                                    jet_E[jet_idx],jet_pt[jet_idx],jet_eta[jet_idx],jet_phi[jet_idx])
            edge_attr.append([dR,dimass])
            if arr['idx_jet_lep_t_b'][i] == jet_idx:
                edge_y.append(1)
            else:
                edge_y.append(0)
            
        ##edge between jets
        for jet1_idx, jet2_idx in itertools.combinations(range(jet_node.shape[0]),2):
            edge_idx.append([jet1_idx,jet2_idx])
            dR = calculate_deltaR(jet_eta[jet1_idx],jet_phi[jet1_idx],jet_eta[jet2_idx],jet_phi[jet2_idx])
            dimass = calculate_mass(jet_E[jet1_idx],jet_pt[jet1_idx],jet_eta[jet1_idx],jet_phi[jet1_idx],
                                    jet_E[jet2_idx],jet_pt[jet2_idx],jet_eta[jet2_idx],jet_phi[jet2_idx])
            edge_attr.append([dR,dimass])
            
            if set([jet1_idx,jet2_idx]).issubset(set([arr['idx_jet_had_t_b'][i],arr['idx_jet_w_u'][i],arr['idx_jet_w_d'][i]])):
                edge_y.append(1)
            else:
                edge_y.append(0)
        
        edge_idx = np.array(edge_idx)
        edge_y = np.array(edge_y)
        edge_attr = np.array(edge_attr)
        
        
        graph_dict = {'lep_node':lepton_node,'met_node':met_node,'jet_node':jet_node,'edge_idx':edge_idx,'edge_attr':edge_attr,'edge_y':edge_y,'weight':arr['weight'][i]}
        graph_arr.append(graph_dict)
        
    graph_arr = np.array(graph_arr)
    n = file.split('_')[-1].replace('.root','')
    np.save(f'{out_path}/graph_{n}.npy',graph_arr)


  
if __name__ == '__main__':

  


  # Define the available working modes

  # Create an argument parser
  parser = argparse.ArgumentParser()

  # Add an argument to select the working mode
  parser.add_argument('--isCondor', action='store_true', default="")
  parser.add_argument('--input_file' ,dest='input_file', type=str,  default="")
  parser.add_argument('--out_path' ,dest='out_path', type=str,  default="")
  # Parse the arguments from the command line
  args = parser.parse_args()
  
  input_file = args.input_file
  isCondor = args.isCondor
  out_path = args.out_path
  
  if isCondor:
    makeGraph(input_file,out_path)
  else:
    files = [os.path.join(input_file,f) for f in os.listdir(input_file)]
    for i,file in enumerate(files):
      #infer(file,input_model_path)
      job = htcondor.Submit({
          "universe":"vanilla",
          "getenv": True,
          "jobbatchname": f"Vcb_Make_Graph",
          "executable": "/data6/Users/yeonjoon/VcbMVAStudy/GNN_ttbar_reco/make_graph.sh",
          "arguments": f" {file} {out_path}",
          "output": f"/data6/Users/yeonjoon/VcbMVAStudy/GNN_ttbar_reco/log/{file.replace('/','_')}.out",
          "error": f"/data6/Users/yeonjoon/VcbMVAStudy/GNN_ttbar_reco/log/{file.replace('/','_')}.err",
          "request_memory": '2GB',
          "should_transfer_files":"YES",
          "when_to_transfer_output" : "ON_EXIT",
      })
    
      schedd = htcondor.Schedd()
      with schedd.transaction() as txn:
        cluster_id = job.queue(txn)
      print("Job submitted with cluster ID:", cluster_id)