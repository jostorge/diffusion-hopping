from rdkit import Chem


def largest_component(molecules):
    return [
        max(
            Chem.GetMolFrags(mol, asMols=True),
            key=lambda x: x.GetNumAtoms(),
            default=mol,
        )
        for mol in molecules
    ]
