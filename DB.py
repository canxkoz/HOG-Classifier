# -*- coding: utf-8 -*-
import pandas as pd
import os

DB_dir = 'images'
DB_csv = 'data.csv'


class Database(object):

  def __init__(self):
    self._gen_csv()
    self.data = pd.read_csv(DB_csv)
    self.classes = set(self.data["group_id"])

  def _gen_csv(self):
    if os.path.exists(DB_csv):
      return
    with open(DB_csv, 'w', encoding='UTF-8') as f:
      f.write("img_loc,group_id,img_id")
      for root, _, files in os.walk(DB_dir, topdown=False):
        for name in files:
          if not name.endswith('.jpg'):
            continue
          group_id = name[:4]
          img_id = name[4:-3]
          img = os.path.join(root, name)
          print('image path: ', img)
          f.write("\n{},{},{}".format(img, group_id, img_id))

  def __len__(self):
    return len(self.data)

  def get_class(self):
    return self.classes

  def get_data(self):
    return self.data


if __name__ == "__main__":
  db = Database()
  data = db.get_data()
  classes = db.get_class()

  print("DB length:", len(db))
  print(classes)
