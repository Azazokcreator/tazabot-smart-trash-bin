#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"

// ===== Wi-Fi =====
const char* WIFI_SSID = "HUAWEI-4QhR";
const char* WIFI_PASS = "fptq48PY";

// ===== AI Thinker pins =====
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

static httpd_handle_t server = NULL;

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static esp_err_t index_handler(httpd_req_t *req) {
  const char* html =
    "<!doctype html><html><head><meta charset='utf-8'/>"
    "<title>ESP32-CAM</title></head><body>"
    "<h2>ESP32-CAM MJPEG Stream</h2>"
    "<p><a href='/stream'>/stream</a> | <a href='/capture'>/capture</a></p>"
    "<img src='/stream' style='max-width:100%;height:auto;'/>"
    "</body></html>";
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, html, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t capture_handler(httpd_req_t *req) {
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }
  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  esp_err_t res = httpd_resp_send(req, (const char*)fb->buf, fb->len);
  esp_camera_fb_return(fb);
  return res;
}

static esp_err_t stream_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  while (true) {
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) continue;

    uint8_t * jpg_buf = fb->buf;
    size_t jpg_len = fb->len;
    bool converted = false;

    if (fb->format != PIXFORMAT_JPEG) {
      bool ok = frame2jpg(fb, 80, &jpg_buf, &jpg_len);
      esp_camera_fb_return(fb);
      fb = NULL;
      if (!ok) continue;
      converted = true;
    }

    // boundary
    if (httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY)) != ESP_OK) {
      if (converted) free(jpg_buf); else if (fb) esp_camera_fb_return(fb);
      break;
    }

    char part_buf[64];
    int hlen = snprintf(part_buf, sizeof(part_buf), STREAM_PART, (unsigned int)jpg_len);
    if (httpd_resp_send_chunk(req, part_buf, hlen) != ESP_OK) {
      if (converted) free(jpg_buf); else if (fb) esp_camera_fb_return(fb);
      break;
    }

    if (httpd_resp_send_chunk(req, (const char*)jpg_buf, jpg_len) != ESP_OK) {
      if (converted) free(jpg_buf); else if (fb) esp_camera_fb_return(fb);
      break;
    }

    if (converted) free(jpg_buf);
    else esp_camera_fb_return(fb);

    // небольшой задержкой иногда повышает стабильность
    // delay(1);
  }

  return ESP_OK;
}

void start_webserver() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.max_uri_handlers = 8;

  if (httpd_start(&server, &config) == ESP_OK) {
    httpd_uri_t uri_index   = { .uri = "/",       .method = HTTP_GET, .handler = index_handler,   .user_ctx = NULL };
    httpd_uri_t uri_capture = { .uri = "/capture",.method = HTTP_GET, .handler = capture_handler, .user_ctx = NULL };
    httpd_uri_t uri_stream  = { .uri = "/stream", .method = HTTP_GET, .handler = stream_handler,  .user_ctx = NULL };

    httpd_register_uri_handler(server, &uri_index);
    httpd_register_uri_handler(server, &uri_capture);
    httpd_register_uri_handler(server, &uri_stream);
  }
}

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(false);

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // стартовые настройки (быстро и стабильно)
  config.frame_size   = FRAMESIZE_QVGA; // 320x240
  config.jpeg_quality = 12;
  config.fb_count     = 2;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    while (true) delay(1000);
  }

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  Serial.print("WiFi connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());

  start_webserver();

  Serial.println("Open:");
  Serial.print("  http://"); Serial.print(WiFi.localIP()); Serial.println("/");
  Serial.print("  http://"); Serial.print(WiFi.localIP()); Serial.println("/stream");
  Serial.print("  http://"); Serial.print(WiFi.localIP()); Serial.println("/capture");
}

void loop() {
  delay(1000);
}
