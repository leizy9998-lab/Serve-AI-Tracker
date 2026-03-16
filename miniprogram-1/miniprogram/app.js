// miniprogram/app.js
App({
  onLaunch: function () {
    this.globalData = {
      env: "serve-2grpc8fr0a87c01e", // ✅ 填这里
    };

    if (!wx.cloud) {
      console.error("请使用 2.2.3 或以上的基础库以使用云能力");
    } else {
      wx.cloud.init({
        env: this.globalData.env,
        traceUser: true,
      });
    }
  }
});
