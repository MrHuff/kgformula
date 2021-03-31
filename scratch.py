import shutil
import os
#
# def forceMergeFlatDir(srcDir, dstDir):
#     if not os.path.exists(dstDir):
#         os.makedirs(dstDir)
#     for item in os.listdir(srcDir):
#         srcFile = os.path.join(srcDir, item)
#         dstFile = os.path.join(dstDir, item)
#         forceCopyFile(srcFile, dstFile)
#
# def forceCopyFile (sfile, dfile):
#     if os.path.isfile(sfile):
#         shutil.copy2(sfile, dfile)
#
# def isAFlatDir(sDir):
#     for item in os.listdir(sDir):
#         sItem = os.path.join(sDir, item)
#         if os.path.isdir(sItem):
#             return False
#     return True
#
#
# def copyTree(src, dst):
#     for item in os.listdir(src):
#         s = os.path.join(src, item)
#         d = os.path.join(dst, item)
#         if os.path.isfile(s):
#             if not os.path.exists(dst):
#                 os.makedirs(dst)
#             forceCopyFile(s,d)
#         if os.path.isdir(s):
#             isRecursive = not isAFlatDir(s)
#             if isRecursive:
#                 copyTree(s, d)
#             else:
#                 forceMergeFlatDir(s, d)
# if __name__ == '__main__':
#     dst = 'base_jobs_kc_layers=3_width=32'
#     src = 'base_jobs_kc_est_rest_layers=3_width=32'
#     ls = os.listdir('do_null_100')
#     for el in ls:
#         try:
#             dst_tmp = f'do_null_100/{el}/{dst}'
#             src_tmp = f'do_null_100/{el}/{src}'
#             copyTree(src_tmp,dst_tmp)
#         except Exception as e:
#             print(e)
if __name__ == '__main__':
    ls = os.listdir('base_jobs_kc_est_bug_rerun')
    print(len(ls))