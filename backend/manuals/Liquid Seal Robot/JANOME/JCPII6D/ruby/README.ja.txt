* Ruby�Ƃ�

Ruby�̓V���v�������͂ȃI�u�W�F�N�g�w���X�N���v�g����ł��D
Ruby�͍ŏ����珃���ȃI�u�W�F�N�g�w������Ƃ��Đ݌v����Ă���
������C�I�u�W�F�N�g�w���v���O���~���O����y�ɍs�������o����
���D�������ʏ�̎葱���^�̃v���O���~���O���\�ł��D

Ruby�̓e�L�X�g�����֌W�̔\�͂ȂǂɗD��CPerl�Ɠ������炢����
�ł��D����ɃV���v���ȕ��@�ƁC��O������C�e���[�^�Ȃǂ̋@�\
�ɂ���āC��蕪����₷���v���O���~���O���o���܂��D


* Ruby�̓���

  + �V���v���ȕ��@
  + ���ʂ̃I�u�W�F�N�g�w���@�\(�N���X�C���\�b�h�R�[���Ȃ�)
  + ����ȃI�u�W�F�N�g�w���@�\(Mixin, ���ك��\�b�h�Ȃ�)
  + ���Z�q�I�[�o�[���[�h
  + ��O�����@�\
  + �C�e���[�^�ƃN���[�W��
  + �K�[�x�[�W�R���N�^
  + �_�C�i�~�b�N���[�f�B���O (�A�[�L�e�N�`���ɂ��)
  + �ڐA���������D������UNIX��œ��������łȂ��CDOS��Windows�C
    Mac�CBeOS�Ȃǂ̏�ł�����


* ����@

** FTP��

�ȉ��̏ꏊ�ɂ����Ă���܂��D

  ftp://ftp.ruby-lang.org/pub/ruby/

** Subversion��

�{�u�����`��Ruby�̍ŐV�̃\�[�X�R�[�h�͎��̃R�}���h�Ŏ擾�ł��܂��D

  $ svn co http://svn.ruby-lang.org/repos/ruby/branches/ruby_1_8/

�J����[�̃\�[�X�R�[�h�͎��̃R�}���h�Ŏ擾�ł��܂��D

  $ svn co http://svn.ruby-lang.org/repos/ruby/trunk/ ruby

���ɊJ�����̃u�����`�̈ꗗ�͎��̃R�}���h�Ō����܂��D

  $ svn ls http://svn.ruby-lang.org/repos/ruby/branches/


* �z�[���y�[�W

Ruby�̃z�[���y�[�W��URL��

   http://www.ruby-lang.org/

�ł��D


* ���[�����O���X�g

Ruby�̃��[�����O���X�g������܂��B�Q����]�̕���

   ruby-list-ctl@ruby-lang.org

�܂Ŗ{����

   subscribe YourFirstName YourFamilyName
   
�Ə����đ����ĉ������B 

Ruby�J���Ҍ������[�����O���X�g������܂��B������ł�ruby�̃o
�O�A�����̎d�l�g���Ȃǎ�����̖��ɂ��ċc�_����Ă��܂��B
�Q����]�̕���

   ruby-dev-ctl@ruby-lang.org

�܂�ruby-list�Ɠ��l�̕��@�Ń��[�����Ă��������B 

Ruby�g�����W���[���ɂ��Ęb������ruby-ext���[�����O���X�g��
���w�֌W�̘b��ɂ��Ęb������ruby-math���[�����O���X�g��
�p��Řb������ruby-talk���[�����O���X�g������܂��B�Q�����@
�͂ǂ�������ł��B 


* �R���p�C���E�C���X�g�[��

�ȉ��̎菇�ōs���Ă��������D

  1. ����configure�t�@�C����������Ȃ��A��������
     configure.in���Â��悤�Ȃ�Aautoconf�����s����
     �V����configure�𐶐�����

  2. configure�����s����Makefile�Ȃǂ𐶐�����

     ���ɂ���Ă̓f�t�H���g��C�R���p�C���p�I�v�V�������t��
     �܂��Dconfigure�I�v�V������ optflags=.. warnflags=.. ��
     �ŏ㏑���ł��܂��D

  3. (�K�v�Ȃ��)defines.h��ҏW����

     �����C�K�v�����Ǝv���܂��D

  4. (�K�v�Ȃ��)ext/Setup�ɐÓI�Ƀ����N����g�����W���[����
     �w�肷��

     ext/Setup�ɋL�q�������W���[���͐ÓI�Ƀ����N����܂��D

     �_�C�i�~�b�N���[�f�B���O���T�|�[�g���Ă��Ȃ��A�[�L�e�N
     �`���ł�Setup��1�s�ڂ́uoption nodynamic�v�Ƃ����s�̃R
     �����g���O���K�v������܂��D�܂��C���̃A�[�L�e�N�`����
     �g�����W���[���𗘗p���邽�߂ɂ́C���炩���ߐÓI�Ƀ���
     �N���Ă����K�v������܂��D

  5. make�����s���ăR���p�C������

  6. make test�Ńe�X�g���s���D

     �utest succeeded�v�ƕ\�������ΐ����ł��D�������e�X�g
     �ɐ������Ă��������ƕۏ؂���Ă����ł͂���܂���D

  7. make install

     root�ō�Ƃ���K�v�����邩������܂���D

�����C�R���p�C�����ɃG���[�����������ꍇ�ɂ̓G���[�̃��O�ƃ}
�V���COS�̎�ނ��܂ނł��邾���ڂ������|�[�g����҂ɑ����Ă�
������Ƒ��̕��̂��߂ɂ��Ȃ�܂��D


* �ڐA

UNIX�ł����configure���قƂ�ǂ̍��ق��z�����Ă����͂���
�����C�v��ʌ����Ƃ����������ꍇ(����ɈႢ�Ȃ�)�C��҂ɂ���
���Ƃ����|�[�g����΁C�����ł��邩���m��܂���D

�A�[�L�e�N�`���ɂ����Ƃ��ˑ�����̂�GC���ł��DRuby��GC�͑Ώ�
�̃A�[�L�e�N�`����setjmp()�ɂ���đS�Ẵ��W�X�^�� jmp_buf��
�i�[���邱�ƂƁCjmp_buf�ƃX�^�b�N��32bit�A���C�������g�����
���邱�Ƃ����肵�Ă��܂��D���ɑO�҂��������Ȃ��ꍇ�̑Ή��͔�
��ɍ���ł��傤�D��҂̉����͔�r�I�ȒP�ŁCgc.c�ŃX�^�b�N��
�}�[�N���Ă��镔���ɃA���C�������g�̃o�C�g���������炵�ă}�[
�N����R�[�h��ǉ����邾���ōς݂܂��D�udefined(THINK_C)�v��
�����Ă��镔�����Q�l�ɂ��Ă�������

# ���ۂɂ�Ruby��Think C�ł̓R���p�C���ł��܂���D

���W�X�^�E�B���h�E������CPU�ł́C���W�X�^�E�B���h�E���X�^�b
�N�Ƀt���b�V������A�Z���u���R�[�h��ǉ�����K�v�����邩���m
��܂���D


* �z�z����

COPYING.ja�t�@�C�����Q�Ƃ��Ă��������B


* ����

�R�����g�C�o�O���|�[�g���̑��� matz@netlab.jp �܂ŁD
-------------------------------------------------------
created at: Thu Aug  3 11:57:36 JST 1995
Local variables:
mode: indented-text
end:
