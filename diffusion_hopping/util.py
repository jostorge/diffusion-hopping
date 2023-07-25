from openbabel import openbabel
from rdkit import RDLogger


def disable_obabel_and_rdkit_logging():
    RDLogger.DisableLog("rdApp.*")
    openbabel.obErrorLog.SetOutputLevel(0)
    openbabel.obErrorLog.StopLogging()
    message_handler = openbabel.OBMessageHandler()
    message_handler.SetOutputLevel(0)
