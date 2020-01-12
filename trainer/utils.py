"""Utils for the models in ML Platform."""

import glob
import os


def upload_local_directory_to_gcs(local_path, bucket, gcs_path):
  """Upload a local directory, recursively, to GCS.

  Args:
   local_path: String with the local directory to be uploaded.
   bucket: Destination bucket object created with a GCS client
   gcs_path: Path in the bucket for the destination directory

  """
  assert os.path.isdir(local_path)
  for local_file in glob.glob(local_path + '/**'):
    if not os.path.isfile(local_file):
      upload_local_directory_to_gcs(
          local_file,
          bucket,
          gcs_path + "/" + os.path.basename(local_file))
    else:
      remote_path = os.path.join(gcs_path, local_file[1 + len(local_path):])
      blob = bucket.blob(remote_path)
      blob.upload_from_filename(local_file)
