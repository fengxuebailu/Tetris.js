import tkinter as tk
from tkinter import messagebox

def main():
    root = tk.Tk()
    root.title("图形界面测试")
    root.geometry("300x200")
    
    label = tk.Label(root, text="如果您能看到这个窗口，说明图形界面可以正常工作")
    label.pack(pady=20)
    
    button = tk.Button(root, text="确定", command=root.quit)
    button.pack()
    
    root.mainloop()

if __name__ == "__main__":
    try:
        main()
        print("窗口已正常显示")
    except Exception as e:
        print(f"错误: {e}")
