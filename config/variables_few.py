from python.variable import Variable

variables = []
variables.append(Variable("dimuon_mass", r"$m_{\mu\mu}$ [GeV]", 50, 110, 150))
variables.append(Variable("LHEMass", r"$m_{genleplep}$ [GeV]", 50, 110, 150))
variables.append(Variable("dimuon_pt", r"$p_{T}(\mu\mu)$ [GeV]", 30, 0, 300))
variables.append(Variable("dimuon_pt_log", r"$log(p_{T}(\mu\mu))$", 50, 0, 200))
variables.append(Variable("dimuon_eta", r"$\eta (\mu\mu)$", 50, -5, 5))
variables.append(Variable("dimuon_phi", r"$\phi (\mu\mu)$", 50, -3.2, 3.2))
variables.append(Variable("dimuon_dEta", r"$\Delta\eta (\mu\mu)$", 50, 0, 10))
variables.append(Variable("dimuon_dPhi", r"$\Delta\phi (\mu\mu)$", 50, 0, 4))
variables.append(Variable("dimuon_dR", r"$\Delta R (\mu\mu)$", 50, 0, 4))
variables.append(Variable("mu1_pt", r"$p_{T}(\mu_{1})$ [GeV]", 30, 0, 300))
variables.append(Variable("mu1_eta", r"$\eta (\mu_{1})$", 50, -2.5, 2.5))
variables.append(Variable("mu1_phi", r"$\phi (\mu_{1})$", 50, -3.2, 3.2))
variables.append(Variable("mu2_pt", r"$p_{T}(\mu_{2})$ [GeV]", 30, 0, 300))
variables.append(Variable("mu2_eta", r"$\eta (\mu_{2})$", 50, -2.5, 2.5))
variables.append(Variable("mu2_phi", r"$\phi (\mu_{2})$", 50, -3.2, 3.2))
variables.append(Variable("genwgt", "genwgt", 50, -100, 1000))
variables.append(Variable("event", "event", 100, 0, 10000000))
variables.append(Variable("run", "run", 1, 0, 1))
variables_lookup = {}
for v in variables:
    variables_lookup[v.name] = v
