import ROOT, tqdm

def split_root_file(input_file, output_prefix, n_files):
    # Open the input file
    file = ROOT.TFile.Open(input_file)

    # Get the TTree from the input file
    tree = file.Get("GNN_tree")

    # Get the total number of entries in the tree
    n_entries = tree.GetEntries()

    # Calculate the number of entries per output file
    entries_per_file = n_entries // n_files

    # Loop over the number of output files
    for i in tqdm.tqdm(range(n_files)):
        # Calculate the range of entries for the current output file
        start_entry = i * entries_per_file
        end_entry = start_entry + entries_per_file - 1

        # Create the output file name
        output_file = f"{output_prefix}_{i}.root"

        # Create a new output file
        output = ROOT.TFile(output_file, "RECREATE")

        # Copy the subset of the tree to the output file
        output_tree = tree.CopyTree("", "", entries_per_file, start_entry)

        # Write the output tree to the output file
        output_tree.Write()

        # Close the output file
        output.Close()

    # Close the input file
    file.Close()
split_root_file('/data6/Users/yeonjoon/SKFlatOutput/Run2UltraLegacy_v3/Vcb/2017/RunGNNTree__RunMu__/Vcb_TTLJ_WtoCB_powheg.root','/data6/Users/yeonjoon/SKFlatOutput/Run2UltraLegacy_v3/Vcb/2017/RunGNNTree__RunMu__/Splitted/Vcb_TTLJ_WtoCB_powheg',1000)