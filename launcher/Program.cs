using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

namespace MingchaoJBLauncher
{
    internal static class Program
    {
        [STAThread]
        private static void Main(string[] args)
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string pythonw = Path.Combine(baseDir, ".venv", "Scripts", "pythonw.exe");
            string python = Path.Combine(baseDir, ".venv", "Scripts", "python.exe");
            string mainPy = Path.Combine(baseDir, "main.py");

            if (!File.Exists(mainPy))
            {
                MessageBox.Show(
                    "找不到主程序 main.py，请确认软件目录完整。\n路径: " + mainPy,
                    "MingchaoJB 启动失败",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return;
            }

            string exePath = File.Exists(pythonw) ? pythonw : (File.Exists(python) ? python : "pythonw.exe");
            string extraArgs = args.Length > 0 ? " " + string.Join(" ", args) : "";

            var psi = new ProcessStartInfo
            {
                FileName = exePath,
                Arguments = "\"" + mainPy + "\"" + extraArgs,
                WorkingDirectory = baseDir,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            try
            {
                Process.Start(psi);
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "启动鸣潮自动任务助手失败：\n" + ex.Message,
                    "MingchaoJB 启动失败",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
            }
        }
    }
}
