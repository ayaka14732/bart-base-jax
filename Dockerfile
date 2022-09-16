FROM archlinux:latest
RUN pacman -Syu --noconfirm python python-pip && \
pacman -Sc --noconfirm && \
useradd --create-home app
USER app
WORKDIR /home/app/app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8080
CMD [ "python", "6_service.py" ]
