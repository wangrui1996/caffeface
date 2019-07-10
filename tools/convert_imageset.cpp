// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/thread.hpp"
#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_int32(num_threads, 4, "number of thread to handle datasets");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "jpg",
    "Optional: What type should we encode the image as ('png','jpg',...).");

shared_ptr<db::Transaction> txn;
// Create new DB
shared_ptr<db::DB> database;
int count = 0;
boost::mutex mu_count;
boost::mutex mu_txn;

struct Parameter {
    std::vector<std::pair<std::string, int> > lines_;
    string encode_type_;
    bool encoded_;
    string root_folder_;
    int resize_height_;
    int resize_width_;
    bool is_color_;
    bool check_size_;
    bool data_size_initialized_;
};

void func_thread(Parameter parameter) {
    Datum datum;
    int lines_size = parameter.lines_.size();
    while (true) {
        mu_count.lock();
        int line_id = count;
        count ++;
        mu_count.unlock();
        if (line_id >= lines_size)
            return;
        bool status;
        std::string enc = parameter.encode_type_;
        if (parameter.encoded_ && !enc.size()) {
            // Guess the encoding type from the file name
            string fn = parameter.lines_[line_id].first;
            size_t p = fn.rfind('.');
            if ( p == fn.npos )
                LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
            enc = fn.substr(p+1);
            std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
        }
        status = ReadImageToDatum(parameter.root_folder_ + parameter.lines_[line_id].first,
                parameter.lines_[line_id].second, parameter.resize_height_, parameter.resize_width_, parameter.is_color_,
                enc, &datum);


        if (!status) continue;
        if (parameter.check_size_) {
            int data_size=0;
            if (!parameter.data_size_initialized_) {
                data_size = datum.channels() * datum.height() * datum.width();
                parameter.data_size_initialized_ = true;
            } else {
                const std::string& data = datum.data();
                CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
                << data.size();
            }
        }
        // sequential
        string key_str = caffe::format_int(line_id, 8) + "_" + parameter.lines_[line_id].first;
        // Put in db
        string out;
        CHECK(datum.SerializeToString(&out));
        mu_txn.lock();
        txn->Put(key_str, out);
        mu_txn.unlock();

    if (line_id % 1000 == 0) {
      // Commit db
      mu_txn.lock();
      txn->Commit();
      txn.reset(database->NewTransaction());
      LOG(INFO) << "Processed " << line_id << " files.";
      mu_txn.unlock();
    }
  }
}

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;

  std::ifstream infile(argv[2]);
  std::vector<std::pair<std::string, int> > lines;
  std::string line;
  size_t pos;
  int label;
  while (std::getline(infile, line)) {
    pos = line.find_last_of(' ');
    label = atoi(line.substr(pos + 1).c_str());
    lines.push_back(std::make_pair(line.substr(0, pos), label));
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  database.reset(db::GetDB(FLAGS_backend));
  database->Open(argv[3], db::NEW);
  txn.reset(database->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);
  Datum datum;
  int data_size = 0;
  bool data_size_initialized = false;

  boost::thread_group tg;

  Parameter parameter;
  parameter.lines_ = lines;
  parameter.encode_type_ = encode_type;
  parameter.encoded_ = encoded;
  parameter.root_folder_ = root_folder;
  parameter.resize_height_ = resize_height;
  parameter.resize_width_ = resize_width;
  parameter.is_color_ = is_color;
  parameter.check_size_ = check_size;
  parameter.data_size_initialized_ = data_size_initialized;

  for (int i = 0; i < FLAGS_num_threads; ++i)
      tg.create_thread(boost::bind(func_thread, parameter));
  tg.join_all();

  // write the last batch
  if (lines.size() % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
