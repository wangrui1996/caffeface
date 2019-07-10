#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace caffe {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

#ifdef USE_OPENCV
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin;
  try {
       cv_img_origin = cv::imread(filename, cv_read_flag);
  } catch (int x) {
     LOG(INFO) << "read image from '" << filename << "' failed"  << std::endl;
  }
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }
  return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width) {
  return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color) {
  return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
  return ReadImageToCVMat(filename, 0, 0, true);
}

// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
                     std::string en) {
  size_t p = fn.rfind('.');
  std::string ext;
        try {
        ext = p != fn.npos ? fn.substr(p+1) : fn;
        } catch (int x) {
            std::cout << "image read fialed: " << fn << std::endl;
            exit(0);
        }


  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);
  if ( ext == en )
    return true;
  if ( en == "jpg" && ext == "jpeg" )
    return true;
  return false;
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          matchExt(filename, encoding) )
        return ReadFileToDatum(filename, label, datum);
      std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      datum->set_label(label);
      datum->set_encoded(true);
      return true;
    }
    CVMatToDatum(cv_img, datum);
    datum->set_label(label);
    return true;
  } else {
    return false;
  }
}

bool ReadImageToDatumPair(const string& filename_first, const string& filename_last,
        const int label, const int height, const int width, const bool is_color,
        const std::string & encoding, DatumPair* datumpair) {
  cv::Mat cv_img_first = ReadImageToCVMat(filename_first, height, width, is_color);
  cv::Mat cv_img_last = ReadImageToCVMat(filename_last, height, width, is_color);
  if (cv_img_first.data) {
    if (encoding.size()) {
      if ( (cv_img_first.channels() == 3) == is_color && !height && !width &&
          matchExt(filename_first, encoding) )
        return ReadFileToDatumPair(filename_first, filename_last, label, datumpair);
      std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img_first, buf);
      datumpair->set_data_first(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      std::vector<uchar> buf1;
      cv::imencode("."+encoding, cv_img_last, buf1);
      datumpair->set_data_last(std::string(reinterpret_cast<char*>(&buf1[0]),
                      buf1.size()));
      datumpair->set_label(label);
      datumpair->set_encoded(true);
      return true;
    }
    CVMatToDatumPair(cv_img_first, cv_img_last, datumpair);
    datumpair->set_label(label);
    return true;
  } else {
    return false;
  }
}
#endif  // USE_OPENCV

bool ReadFileToDatum(const string& filename, const int label,
    Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    datum->set_label(label);
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}

bool ReadFileToDatumPair(const string& filename_first, const string& filename_last,
        const int label, DatumPair* datumPair) {
  std::streampos size;

  fstream file_first(filename_first.c_str(), ios::in|ios::binary|ios::ate);
  fstream file_last(filename_last.c_str(), ios::in|ios::binary|ios::ate);
  if (file_first.is_open() && file_last.is_open()) {
    size = file_first.tellg();
    std::string buffer(size, ' ');
    file_first.seekg(0, ios::beg);
    file_first.read(&buffer[0], size);
    file_first.close();
    datumPair->set_data_first(buffer);
    size = file_last.tellg();
    std::string buffer1(size, ' ');
    file_last.seekg(0, ios::beg);
    file_last.read(&buffer1[0], size);
    file_last.close();
    datumPair->set_data_last(buffer1);
    datumPair->set_label(label);
    datumPair->set_encoded(true);
    return true;
  } else {
    return false;
  }
}

#ifdef USE_OPENCV
cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  cv_img = cv::imdecode(vec_data, -1);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}
std::pair<cv::Mat,cv::Mat> DecodeDatumPairToCVMatNative(const DatumPair& datumpair) {
        cv::Mat cv_img_first;
        CHECK(datumpair.encoded()) << "Datum not encoded";
        const string& data_first = datumpair.data_first();
        std::vector<char> vec_data_first(data_first.c_str(), data_first.c_str() + data_first.size());
        cv_img_first = cv::imdecode(vec_data_first, -1);
        if (!cv_img_first.data) {
            LOG(ERROR) << "Could not decode datum ";
        }

        cv::Mat cv_img_last;
        CHECK(datumpair.encoded()) << "Datum not encoded";
        const string& data_last = datumpair.data_last();
        std::vector<char> vec_data_last(data_last.c_str(), data_last.c_str() + data_last.size());
        cv_img_last = cv::imdecode(vec_data_last, -1);
        if (!cv_img_last.data) {
            LOG(ERROR) << "Could not decode datum ";
        }

        return std::make_pair(cv_img_first, cv_img_last);
    }

cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}

std::pair<cv::Mat, cv::Mat> DecodeDatumPairToCVMat(const DatumPair& datumpair, bool is_color) {
    cv::Mat cv_img_first;
    CHECK(datumpair.encoded()) << "Datum not encoded";
    const string& data = datumpair.data_first();
    std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
    int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
                            CV_LOAD_IMAGE_GRAYSCALE);
    cv_img_first = cv::imdecode(vec_data, cv_read_flag);
    if (!cv_img_first.data) {
            LOG(ERROR) << "Could not decode datum ";
    }
    cv::Mat cv_img_last;
    CHECK(datumpair.encoded()) << "Datum not encoded";
    const string& data_last = datumpair.data_last();
    std::vector<char> vec_data_last(data_last.c_str(), data_last.c_str() + data_last.size());
    cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
                        CV_LOAD_IMAGE_GRAYSCALE);
    cv_img_last = cv::imdecode(vec_data_last, cv_read_flag);
    if (!cv_img_last.data) {
        LOG(ERROR) << "Could not decode datum ";
    }

    return std::make_pair(cv_img_first, cv_img_last);
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}
bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}

void CVMatToDatumPair(const cv::Mat& cv_img_first, const cv::Mat& cv_img_last, DatumPair* datumpair) {
  CHECK(cv_img_first.depth() == CV_8U) << "Image data type must be unsigned byte";
  CHECK(cv_img_last.depth() == CV_8U) << "Image data type must be unsigned byte";
  datumpair->set_channels(cv_img_first.channels());
  datumpair->set_height(cv_img_first.rows);
  datumpair->set_width(cv_img_first.cols);
  datumpair->clear_data_first();
  datumpair->clear_data_last();
  datumpair->clear_float_data_first();
  datumpair->clear_float_data_last();
  datumpair->set_encoded(false);
  int datum_channels = datumpair->channels();
  int datum_height = datumpair->height();
  int datum_width = datumpair->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img_first.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datumpair->set_data_first(buffer);
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img_last.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datumpair->set_data_last(buffer);
}

#endif  // USE_OPENCV
}  // namespace caffe
