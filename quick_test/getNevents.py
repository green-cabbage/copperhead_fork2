import glob
import ROOT as rt
import concurrent.futures

def getNevents(fname):
    try:

        file = rt.TFile(fname)
        return file["Events"].GetBranch("event").GetEntries()
    except Exception as e:
        return f"An error occurred with file {fname}: {e}"


fileListRERECO_private = glob.glob("/eos/purdue/store/group/local/hmm/FSRnano18ABC_NANOV10b/SingleMuon/RunIISummer16MiniAODv3_FSRnano18ABC_NANOV10b_un2018A-17Sep2018-v2/*/*/*.root")


with concurrent.futures.ProcessPoolExecutor(max_workers=60) as executor:
    # Submit each file check to the executor
    # results = list(executor.map(getNevents, fileListRERECO_private))
    results = []
    futures = [executor.submit(getNevents, fname) for fname in fileListRERECO_private]
        
    # Collect results as they complete
    for future in concurrent.futures.as_completed(futures):
        results.append(future.result())
print(f"Succes! total nevents: {sum(results)}")
