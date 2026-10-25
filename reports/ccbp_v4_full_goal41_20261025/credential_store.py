"""User-authorized Windows Credential Manager entry for this exact SSH account.

No plaintext credential files; never prints the credential. Not experiment code.
"""
import ctypes
from ctypes import wintypes
import getpass

TARGET='Codex/MLR/jq1.9gpu.com:11920/root'
class CREDENTIAL(ctypes.Structure):
    _fields_=[('Flags',wintypes.DWORD),('Type',wintypes.DWORD),('TargetName',wintypes.LPWSTR),
              ('Comment',wintypes.LPWSTR),('LastWritten',wintypes.FILETIME),
              ('CredentialBlobSize',wintypes.DWORD),('CredentialBlob',ctypes.POINTER(ctypes.c_ubyte)),
              ('Persist',wintypes.DWORD),('AttributeCount',wintypes.DWORD),('Attributes',ctypes.c_void_p),
              ('TargetAlias',wintypes.LPWSTR),('UserName',wintypes.LPWSTR)]

def load():
    api=ctypes.WinDLL('Advapi32.dll',use_last_error=True)
    api.CredReadW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.POINTER(ctypes.POINTER(CREDENTIAL))]
    api.CredReadW.restype=wintypes.BOOL;api.CredFree.argtypes=[ctypes.c_void_p]
    p=ctypes.POINTER(CREDENTIAL)()
    if not api.CredReadW(TARGET,1,0,ctypes.byref(p)):
        if ctypes.get_last_error()==1168:return None
        raise ctypes.WinError(ctypes.get_last_error())
    try:return ctypes.string_at(p.contents.CredentialBlob,p.contents.CredentialBlobSize).decode('utf-16-le')
    finally:api.CredFree(p)

def store(secret):
    api=ctypes.WinDLL('Advapi32.dll',use_last_error=True)
    api.CredWriteW.argtypes=[ctypes.POINTER(CREDENTIAL),wintypes.DWORD];api.CredWriteW.restype=wintypes.BOOL
    raw=secret.encode('utf-16-le');blob=(ctypes.c_ubyte*len(raw)).from_buffer_copy(raw)
    c=CREDENTIAL();c.Type=1;c.TargetName=TARGET;c.UserName='root';c.Persist=2
    c.Comment='Explicit user authorization: reusable MLR experiment SSH credential.'
    c.CredentialBlobSize=len(raw);c.CredentialBlob=ctypes.cast(blob,ctypes.POINTER(ctypes.c_ubyte))
    try:
        if not api.CredWriteW(ctypes.byref(c),0):raise ctypes.WinError(ctypes.get_last_error())
    finally:ctypes.memset(blob,0,len(raw));raw=None

if __name__=='__main__':
    value=getpass.getpass('SSH credential (no echo; Windows Credential Manager): ')
    assert value
    store(value);assert load()==value;value=None
    print('Credential Manager entry stored and verified; secret not printed.')
